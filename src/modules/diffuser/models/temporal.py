import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from einops import rearrange
import pdb
from torch.distributions import Bernoulli
import numpy as np

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine = True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class GlobalMixing(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
            Conv1dBlock(out_channels, out_channels, kernel_size, mish),
        ])

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Encoder_Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_Unet, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled

class Decoder_Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_Unet, self).__init__()
        # self.upconv = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.upconv = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        returns_condition=False,
        condition_dropout=0.1,
        calc_energy=False,
        kernel_size=5,
        mlp_node="mlp2"
    ):
        super().__init__()
        # dim_mults = (1,4,8)
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
            mish = True
            act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim
        self.mlp_mode = mlp_node

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                        nn.Linear(1, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            embed_dim = 2*dim
        else:
            embed_dim = dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(64, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, 1, 1),
        )
        
        self.pre = nn.Linear(cond_dim + transition_dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256),
            act_fn,
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            act_fn,
            nn.Linear(256, transition_dim),
        )
        self.adapt_dim = nn.Linear(dim,transition_dim)
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )
        self.enc1 = Encoder_Unet(1, 64)
        self.enc2 = Encoder_Unet(64, 128)
        self.enc3 = Encoder_Unet(128, 256)
        self.bottleneck = ConvBlock(256, 512)
        self.dec3 = Decoder_Unet(512, 256)
        self.dec2 = Decoder_Unet(256, 128)
        self.dec1 = Decoder_Unet(128, 64)

        self.enc2_1 = Encoder_Unet(1, 8)
        self.enc2_2 = Encoder_Unet(8, 16)
        self.bottleneck2 = ConvBlock(16, 32)
        self.dec2_2 = Decoder_Unet(32, 16)
        self.dec2_1 = Decoder_Unet(16, 8)
        self.final_conv2 = nn.Sequential(
            Conv1dBlock(8, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, 1, 1),
        )
        # self.final_conv = nn.Conv1d(64, transition_dim, kernel_size=1)
    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        ##################################################
        #################为啥这里没用到cond################
        ##################################################
        # x = x.squeeze(1) # 3 154
        # trg_seq = x[:,:-cond.shape[1]] + self.time_mlp_transition(time)
        # trg_seq = trg_seq_all[:,:] + self.time_mlp_transition(time)
        src_seq = torch.cat((x, cond), dim=-1)

        t = self.time_mlp(time) # 3 128
        x = (self.pre(src_seq) + t).unsqueeze(1)

        # x1, x = self.enc1(x)
        # x2, x = self.enc2(x)
        # x3, x = self.enc3(x)
        # x = self.bottleneck(x)
        # x = self.dec3(x, x3)
        # x = self.dec2(x, x2)
        # x = self.dec1(x, x1)
        # x = self.final_conv(x).squeeze(1)

        x1, x = self.enc2_1(x)
        x2, x = self.enc2_2(x)
        x = self.bottleneck2(x)
        x = self.dec2_2(x, x2)
        x = self.dec2_1(x, x1)
        x = self.final_conv2(x).squeeze(1)
 
        return self.adapt_dim(x)


        # return 
        # if self.calc_energy:
        #     x_inp = x

        # x = einops.rearrange(x, 'b h t -> b t h')
        # # x = x.squeeze(1)

        # t = self.time_mlp(time)

        # if self.returns_condition:
        #     assert returns is not None
        #     returns_embed = self.returns_mlp(returns)
        #     if use_dropout:
        #         mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
        #         returns_embed = mask*returns_embed
        #     if force_dropout:
        #         returns_embed = 0*returns_embed
        #     t = torch.cat([t, returns_embed], dim=-1)

        # h = []

        # for resnet, resnet2, downsample in self.downs:
        #     x = resnet(x, t)
        #     x = resnet2(x, t)
        #     h.append(x)
        #     x = downsample(x)

        # x = self.mid_block1(x, t)
        # x = self.mid_block2(x, t)

        # # import pdb; pdb.set_trace()

        # for resnet, resnet2, upsample in self.ups:
        #     x = torch.cat((x, h.pop()), dim=1)
        #     # x = torch.cat((x, h.pop()), dim=0)
        #     x = resnet(x, t)
        #     x = resnet2(x, t)
        #     x = upsample(x)

        # x = self.final_conv(x)

        # x = einops.rearrange(x, 'b t h -> b h t')

        # if self.calc_energy:
        #     # Energy function
        #     energy = ((x - x_inp)**2).mean()
        #     grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
        #     return grad[0]
        # else:
        #     return x
        

        

    def get_pred(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x

class MLPnet(nn.Module):
    def __init__(
        self,
        transition_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        horizon=1,
        returns_condition=True,
        condition_dropout=0.1,
        calc_energy=False,
    ):
        super().__init__()

        if calc_energy:
            act_fn = nn.SiLU()
        else:
            act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy
        self.transition_dim = transition_dim
        self.action_dim = transition_dim - cond_dim

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                        nn.Linear(1, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            embed_dim = 2*dim
        else:
            embed_dim = dim

        self.mlp = nn.Sequential(
                        nn.Linear(embed_dim + transition_dim, 1024),
                        act_fn,
                        nn.Linear(1024, 1024),
                        act_fn,
                        nn.Linear(1024, self.action_dim),
                    )

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x action ]
            cond: [batch x state]
            returns : [batch x 1]
        '''
        # Assumes horizon = 1
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        inp = torch.cat([t, cond, x], dim=-1)
        out  = self.mlp(inp)

        if self.calc_energy:
            energy = ((out - x) ** 2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x, create_graph=True)
            return grad[0]
        else:
            return out

class TemporalValue(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        time_dim=None,
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = time_dim or dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])

        print(in_out)
        for dim_in, dim_out in in_out:

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            horizon = horizon // 2

        fc_dim = dims[-1] * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out














#################################################################
##############################  transformer  ####################
#################################################################
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q x k^T
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # if mask is not None:
        #     # 把mask中为0的数置为-1e9, 用于decoder中的masked self-attention
        #     attn = attn.masked_fill(mask == 0, -1e9)
        
        # dim=-1表示对最后一维softmax
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # 三个线性层做矩阵乘法生成q, k, v.
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        # ScaledDotProductAttention见下方
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # b: batch_size, lq: translation task的seq长度, n: head数, dv: embedding vector length
        # Separate different heads: b x lq x n x dv. 
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # project & reshape
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting. 
            # (batchSize, 1, seqLen) -> (batchSize, 1, 1, seqLen)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # view只能用在contiguous的variable上
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # add & norm
        q += residual

        q = self.layer_norm(q)

        return q, attn
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        # add & norm
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # masked self-attention
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # encoder-decoder attention
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        # q用自己的, k和v是encoder的输出
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        # 将tensor注册成buffer, optim.step()的时候不会更新
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            # 2i, 所以此处要//2. 
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i 偶数
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 奇数

        return torch.FloatTensor(sinusoid_table).unsqueeze(0) # shape:(1, maxLen(n_position), d_hid)

    def forward(self, x):
        # print(x.shape,self.pos_table.shape,self.pos_table[:, :x.size(1)].shape)
        # assert x.shape == self.pos_table.squeeze(0).shape
        # return x + self.pos_table[:, :x.size(1)].clone().detach() # 数据、梯度均无关
        return x.unsqueeze(1) + self.pos_table[:, :x.unsqueeze(1).size(1), :].clone().detach() # 数据、梯度均无关

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200):

        super().__init__()
        # Embedding
        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        #######################################换成了线性层，因为embeding要每个词有不同的idx#############################################
        self.src_word_emb = nn.Sequential(
                        nn.Linear(n_src_vocab, d_word_vec, bias=False),
                        nn.ReLU(),
                    )
        # Position Encoding
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        # 多个Encoder Layer叠加
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        
        # Embedding & Position encoding
        a=self.src_word_emb(src_seq)
        b = self.position_enc(a)
        enc_output = self.dropout(b)
        enc_output = self.layer_norm(enc_output)

        # Encoder Layers
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1):

        super().__init__()
        # ebedding词集变成目标语言词集
        # self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        #####################################换成了线性层########################################
        self.trg_word_emb =  nn.Sequential(
                        nn.Linear(n_trg_vocab, d_word_vec, bias=False),
                        nn.ReLU(),
                    )
        # nn.Linear(n_trg_vocab, d_word_vec, bias=False),
        # self.relu = nn.ReLU(),       
        # Position encoding
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        # 多个Decoder Layer叠加
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward

        # Embedding & Position encoding
        a = self.trg_word_emb(trg_seq)
        dec_output = self.dropout(self.position_enc(a))
        dec_output = self.layer_norm(dec_output)

        # Decoder Layers
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

def get_pad_mask(seq, pad_idx):
    # (batch, seqlen) -> (batch, 1, seqlen) 
    return (seq != pad_idx).unsqueeze(-2)

# 返回一个对角线以下为True的矩阵
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    # torch.triu(diagonal=1)保留矩阵上三角部分，其余部分(包括对角线)定义为0。
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask




class Transformer(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        returns_condition=False,
        condition_dropout=0.1,
        calc_energy=False,
        kernel_size=5,
        mlp_node="mlp2"
    ):
        super().__init__()
        # dim_mults = (1,4,8)
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
            mish = True
            act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim
        self.mlp_mode = mlp_node

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                        nn.Linear(1, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            embed_dim = 2*dim
        else:
            embed_dim = dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
        )
        
        self.pre = nn.Linear(cond_dim + transition_dim, dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Dropout(0.5),
            act_fn,
            nn.Linear(256, 512),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            act_fn,
            nn.Linear(256, transition_dim),
        )   
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, 256),
            act_fn,
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            act_fn,
            nn.Linear(256, transition_dim),
        ) 
        self.mlp3 = nn.Sequential(
            nn.Linear(dim, 256),
            act_fn,
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 256),
            act_fn,
            nn.Linear(256, transition_dim),
        ) 
        self.mlp4 = nn.Sequential(
            nn.Linear(dim, 256),
            act_fn,
            nn.Linear(256, 256),
            act_fn,
            nn.Linear(256, 512),
            act_fn,
            nn.Linear(512, 1024),
            act_fn,
            nn.Linear(1024, 2048),
            act_fn,
            nn.Linear(2048, 1024),
            act_fn,
            nn.Linear(1024, 512),
            act_fn,
            nn.Linear(512, 256),
            act_fn,
            nn.Linear(256, 256),
            act_fn,
            nn.Linear(256, transition_dim),
        )

        self.resnet1 = nn.Sequential(
            nn.Linear(dim, 256),
            act_fn,
            nn.Linear(256, 256),
            act_fn,)
        self.resnet2 = nn.Sequential(
            nn.Linear(256, 512),
            act_fn,
            nn.Linear(512, 256),
            act_fn,)
        
        self.resnet3 = nn.Sequential(
            nn.Linear(256, 512),
            act_fn,
            nn.Linear(512, 1024),
            act_fn,
            nn.Linear(1024, 256),
            act_fn,)
        
        self.resnet4 = nn.Sequential(
            nn.Linear(256, 256),
            act_fn,
            nn.Linear(256, 256),
            act_fn,)
        self.resnet5 = nn.Sequential(
            nn.Linear(256, transition_dim),
        )


        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )
    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        ##################################################
        #################为啥这里没用到cond################
        ##################################################
        # x = x.squeeze(1) # 3 154
        # trg_seq = x[:,:-cond.shape[1]] + self.time_mlp_transition(time)
        # trg_seq = trg_seq_all[:,:] + self.time_mlp_transition(time)
        src_seq = torch.cat((x, cond), dim=-1)

        t = self.time_mlp(time) # 3 128
        x_embd = self.pre(src_seq) + t
        if self.mlp_mode == "mlp2":
            return self.mlp2(x_embd)
        elif self.mlp_mode == "mlp3":
            return self.mlp3(x_embd)
        elif self.mlp_mode == "mlp4":
            return self.mlp4(x_embd)
        elif self.mlp_mode == "mlp5":
            x1 = self.resnet1(x_embd)   #256
            x2 = self.resnet2(x1)       #256
            x2 = self.resnet3(x1 + x2)  #256
            x3 = self.resnet4(x2)       #256
            x = self.resnet5(x3 + x2)   #256
            return x
        else:
            raise ValueError('self.mlp_mode must be mlp2, mlp3, mlp4, mlp5.')







# class Transformer(nn.Module):
#     ''' A sequence to sequence model with attention mechanism. '''
#     def __init__(
#             self,   
#             horizon,
#             transition_dim,    #就是n_trg_vocab
#             cond_dim,          #就是n_src_vocab
#             dim=128,
#             dim_mults=(1, 2, 4, 8),
#             returns_condition=False,
#             condition_dropout=0.1,
#             calc_energy=False,
#             kernel_size=5,
#             src_pad_idx=1, trg_pad_idx=0,
#             d_word_vec=256, d_model=256, d_inner=1024,
#             n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=1,
#             trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):

#         super().__init__()

#         self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

#         # Encoder
#         self.encoder = Encoder(
#             # n_src_vocab=cond_dim,
#             n_src_vocab=dim, n_position=n_position,
#             d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
#             n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
#             pad_idx=src_pad_idx, dropout=dropout)

#         # Decoder
#         # self.decoder = Decoder(
#         #     # n_trg_vocab=transition_dim,
#         #     n_trg_vocab=dim, n_position=n_position,
#         #     d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
#         #     n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
#         #     pad_idx=trg_pad_idx, dropout=dropout)

#         # 最后的linear输出层
#         self.trg_word_prj = nn.Linear(d_model, transition_dim, bias=False)

#         # xavier初始化
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p) 

#         assert d_model == d_word_vec, \
#         'To facilitate the residual connections, \
#          the dimensions of all module outputs shall be the same.'

#         self.x_logit_scale = 1.
#         if trg_emb_prj_weight_sharing:
#             # Share the weight between target word embedding & last dense layer
# ###############################注掉了##########################
#             # self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
#             self.x_logit_scale = (d_model ** -0.5)

#         # if emb_src_trg_weight_sharing:
#         #     self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


#         if calc_energy:
#             mish = False
#             act_fn = nn.SiLU()
#         else:
#             mish = True
#             act_fn = nn.Mish()

#         self.time_dim = dim

#         # self.time_mlp_transition = nn.Sequential(
#         #     SinusoidalPosEmb(dim),
#         #     nn.Linear(dim, dim * 4),
#         #     act_fn,
#         #     nn.Linear(dim * 4, transition_dim),
#         # )
#         # self.time_mlp_cond = nn.Sequential(
#         #     SinusoidalPosEmb(dim),
#         #     nn.Linear(dim, dim * 4),
#         #     act_fn,
#         #     nn.Linear(dim * 4, cond_dim),
#         # )

#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(dim),
#             nn.Linear(dim, dim * 4),
#             act_fn,
#             nn.Linear(dim * 4, dim),
#         )
#         self.pre = nn.Sequential(
#             nn.Linear(cond_dim + transition_dim, dim),
#             nn.Sigmoid(),)

#     # def forward(self, trg_seq_all, src_seq, time, returns=None, use_dropout=True, force_dropout=False):
#     def forward(self, x_noisy, cond, time, returns=None, use_dropout=True, force_dropout=False):
#         inputs = torch.cat((x_noisy, cond), dim=-1)

#         # trg_seq = trg_seq_all[:,:] + self.time_mlp_transition(time)
#         src_seq = self.pre(inputs) + self.time_mlp(time)
#         # src_seq = cond + self.time_mlp_cond(time)
#         # trg_seq = x_noisy + self.time_mlp_transition(time)
#         # mask
#         src_mask = get_pad_mask(src_seq, self.src_pad_idx)
#         # trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

#         # encoder & decoder
#         enc_output, *_ = self.encoder(src_seq, src_mask)
#         # dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        
#         # final linear layer得到logit vector
#         seq_logit = self.trg_word_prj(enc_output)
#         seq_logit = seq_logit * self.x_logit_scale
#         return seq_logit.view(-1, seq_logit.size(2))