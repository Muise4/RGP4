import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb

import modules.diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)
# 智能体得到“观测”预测动作，但在我们的实验中，得到观测预测状态。
class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, hidden_dim=256,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1, ar_inv=False, train_only_inv=False):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        if self.ar_inv:
            self.inv_model = ARInvModel(hidden_dim=hidden_dim, observation_dim=observation_dim, action_dim=action_dim)
        else:
            self.inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim),
            )
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        # 输出一个形状与x相同的新张量y，其中每个元素的值等于其在x及之前所有元素的乘积。
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # aqrt计算输入张量每个元素的平方根
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)
        self.loss_MSE = torch.nn.MSELoss()
        self.loss_BCE = torch.nn.BCELoss() #需要加sigmoid
            # self.loss_fn = torch.nn.BCEWithLogitsLoss() #已经带了sigmoid，但是可能会输出负数就很男泵
        self.sigmoid = nn.Sigmoid()
        self.BCE=None
        self.MSE=None

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)
        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#.
    #------------------------------------------- testing -------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        #去噪过程
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        ####################################################3
        ####transformer只输出状态的话就要cat#################
        ###################################################3
        # if x_recon.shape != x.shape:
        #     x_recon = torch.cat((x_recon, cond), dim=-1)
        # assert x.shape == x_recon.shape
        # 这里为什么多了这一步？这里跟视频没什么关系啊
        # clip_denoised:剪辑去噪
        ################################################################
        ###############################################################3
        if self.clip_denoised:
            # clamp是把输出限制在(-1., 1.)内，大于1的变1，小于-1的变-1
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        # b是batchsize,取batchsize * n_agent
        b, *_, device = *x.shape, x.device
        # 预测均值和方差
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        # 取一个和x同样大小的噪声,用冲参数化来做,同时需要一个标准高斯分布,形状为x的标准高斯分布
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        # 扩散出来的模型到了最后一步时就不需要采样了
        # t=0时,nonzero_mask为0,即不采样,t不为0时,调整形状和model_mean一样.
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # 对冲参数化进行采样
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device
        bs_n_agent = shape[0]
        ####################################开始预测####################################
        # 生成噪声，x就是噪声, x:[bs_n_agent,self.action_dim+self.observation_dim]
        x = 0.5*torch.randn(shape, device=device)
        # x的后半部分填充为cond
        # 将x与condition拼接，后面的是condition，前面的是待生成的部分state
        # x = apply_conditioning(x, cond, 0)
        # if return_diffusion: diffusion = [x] 标志着要不要返回去噪过程,文章中图片去噪的那个图
        # 去噪的进度条
        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        # 逆向过程
        ####################################真正预测####################################
        for i in reversed(range(0, self.n_timesteps)):
            # n个agent,构建n个去噪过程,timesteps就是t,去噪过程100步,timesteps就=100
            # [bs_n_agent],值为i
            timesteps = torch.full((bs_n_agent,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            # x = apply_conditioning(x, cond, 0)
            # assert x.shape == shape
            # progress.update({'t': i})

            # if return_diffusion: diffusion.append(x)

        # progress.close()

        # if return_diffusion:
        #     return x, torch.stack(diffusion, dim=1)
        # else:
        return x
    
    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
            这个return是强化学习里面的评分函数的评分,输入的是一个数,训练时是真实拿到的评分,测试时是一个大的数
        '''
        # device = self.betas.device
        # 这里的batchsize指的是啥呢？三个智能体，obs和h的维度合并是106
        # cond是[3,106]的tensor，但这的len为啥取的是106呢
        bs_n_agent = len(cond)
        # horizon预测后续多少步的xstart
        # horizon = horizon or self.horizon
        # shape = (n_agent, horizon, self.observation_dim)
        # self.action_dim是state_shape
        ####################################shape = (bs_n_agent, self.action_dim+self.observation_dim)
        shape = (bs_n_agent, self.action_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)


#------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # extract
        # 提取指定时间步长的一些系数，
        # 然后重新调整为 [batch_size， 1， 1， 1， 1， ...] 以进行广播。
        # 加噪声，然后训练 
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    # def p_losses(self, x_start, cond, t, returns=None):
    #     # 扩一下x_start:[bs*n_agent,rnn_hidden_dim+rnn_hidden_dim+rnn_hidden_dim]
    #     x_start = torch.cat((x_start, cond), dim=-1)
    #     # 生成随机噪声，与x_start形状相同的noise
    #     noise = torch.randn_like(x_start)
    #     # 噪声采样,注意这个是一次性完成的
    #     # 扩散模型的前向过程，x_noisy与x_start.shape同
    #     # x_noisy是向x_start中加了噪声
    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    #     # 把x_noisy[:, -conditions.shape[1]:]用cond填充，并用后面的cond一点点更新前面的噪声成为x_start
    #     x_noisy = apply_conditioning(x_noisy, cond, 0)      
    #     # 由nosiy预测xstart，
    #     # 因为预测整个的xstart，所以返回x_recon前面的一部分，真正的xstart的预测
    #     # 将采样的x和self condition的x一起输入到model当中    
    #     x_recon = self.model(x_noisy, cond, t, returns)
    #     ####################################################3
    #     ####transformer只输出状态的话就要cat#################
    #     ###################################################3
    #     if noise.shape != x_recon.shape:
    #         x_recon = torch.cat((x_recon, cond), dim=-1)
    #     assert noise.shape == x_recon.shape
    #     # 如果不是预测噪声：即如果预测真实情况
    #     if not self.predict_epsilon:
    #         x_recon = apply_conditioning(x_recon, cond, 0)

    def get_BCE_MSE(self, x_start):
            BCE_idx = []
            # MSE_idx = [] 
            # BCE_idx = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ...]
            # MSE_idx = [2, 3, 20, 21, 38, 39, 56, 57, 74, 75, 91, 92, 97, 98, ...]
            for i in range(x_start.shape[1]):
                # 全是整数,BCE
                if x_start[:,i].clone().type(torch.int).sum() == x_start[:,i].sum():
                    BCE_idx.append(i)
                # else:
                #     MSE_idx.append(i)
            BCE = torch.zeros(x_start.shape[1])
            BCE[BCE_idx] = 1
            MSE = 1 - BCE
            # return BCE.cuda(), MSE.cuda()
            return BCE, MSE

    def get_BCE_MSE_list(self, x_start):
            BCE_idx = []
            MSE_idx = [] 
            for i in range(x_start.shape[1]):
                # 全是整数,BCE
                if x_start[:,i].clone().type(torch.int).sum() == x_start[:,i].sum():
                    BCE_idx.append(i)
                else:
                    MSE_idx.append(i)
            return BCE_idx, MSE_idx
    
    def p_losses(self, x_start, cond, t, returns=None):
        # 扩一下x_start:[bs*n_agent,rnn_hidden_dim+rnn_hidden_dim+rnn_hidden_dim]
        # x_start = torch.cat((x_start, cond), dim=-1)
        # 生成随机噪声，与x_start形状相同的noise
        noise = torch.randn_like(x_start)
        # 噪声采样,注意这个是一次性完成的
        # 扩散模型的前向过程，x_noisy与x_start.shape同
        # x_noisy是向x_start中加了噪声
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # 把x_noisy[:, -conditions.shape[1]:]用cond填充，并用后面的cond一点点更新前面的噪声成为x_start
        # x_noisy = apply_conditioning(x_noisy, cond, 0)      
        # 由nosiy预测xstart，
        # 因为预测整个的xstart，所以返回x_recon前面的一部分，真正的xstart的预测
        # 将采样的x和self condition的x一起输入到model当中
        x_recon = self.model(x_noisy, cond, t, returns)
        ####################################################3
        ####transformer只输出状态的话就要cat#################
        ###################################################3
        # if x_noisy.shape != x_recon.shape:
        #     x_recon = torch.cat((x_recon, cond), dim=-1)
        # assert x_noisy.shape == x_recon.shape
        # assert x_recon.shape == noise.shape
        # 如果不是预测噪声：即如果预测真实情况
        # 如果预测噪声
        # if self.predict_epsilon:
        #     # loss, info = self.loss_fn(x_recon[:,:-cond.shape[1]], noise[:,:-cond.shape[1]])
        #     loss = self.loss_MSE(x_recon, noise)

        # else:
        #     if self.BCE == None:
        #         self.BCE, self.MSE = self.get_BCE_MSE(x_start)
        #     if x_recon.device != self.BCE.device:
        #         self.BCE = self.BCE.to(x_recon.device)
        #         self.MSE = self.MSE.to(x_recon.device)
        #     BCE_output = self.sigmoid(x_recon * self.BCE)
        #     BCE_loss = self.loss_BCE(BCE_output, x_start * self.BCE)
        #     MSE_output = x_recon * self.MSE
        #     MSE_loss = self.loss_MSE(MSE_output, x_start * self.MSE)
        #     loss = (len(self.BCE)/x_start.shape[1]) * BCE_loss + (len(self.MSE)/x_start.shape[1]) * MSE_loss
        # return loss
        # # 如果预测噪声
        if self.predict_epsilon:
            # loss, info = self.loss_fn(x_recon[:,:-cond.shape[1]], noise[:,:-cond.shape[1]])
            # loss = self.loss_fn(x_recon[:,:-cond.shape[1]], noise[:,:-cond.shape[1]])
            loss = self.loss_MSE(x_recon, noise)
        else:
            # loss, info = self.loss_fn(x_recon[:,:-cond.shape[1]], x_start[:,:-cond.shape[1]])
            # loss = self.loss_fn(x_recon[:,:-cond.shape[1]], x_start[:,:-cond.shape[1]])
            loss = self.loss_MSE(x_recon, x_start)
        # return x_recon
        return loss

    def loss(self, x, cond, returns=None):
        # if self.train_only_inv:
        #     # Calculating inv loss
        #     x_t = x[:, :-1, self.action_dim:]
        #     a_t = x[:, :-1, :self.action_dim]
        #     x_t_1 = x[:, 1:, self.action_dim:]
        #     x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        #     x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
        #     a_t = a_t.reshape(-1, self.action_dim)
        #     if self.ar_inv:
        #         loss = self.inv_model.calc_loss(x_comb_t, a_t)
        #         info = {'a0_loss':loss}
        #     else:
        #         pred_a_t = self.inv_model(x_comb_t)
        #         loss = F.mse_loss(pred_a_t, a_t)
        #         info = {'a0_loss': loss}
        # else:
            batch_size = len(x)
            # t是维度为[batch_size],[0,n_timesteps]内的随机数
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            # states = self.p_losses(x, cond, t, returns)
            # Calculating inv loss
            # x_t = x[:, :-1, self.action_dim:]
            # a_t = x[:, :-1, :self.action_dim]
            # x_t_1 = x[:, 1:, self.action_dim:]
            # x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            # x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            # a_t = a_t.reshape(-1, self.action_dim)
            # if self.ar_inv:
            #     inv_loss = self.inv_model.calc_loss(x_comb_t, a_t)
            # else:
            #     pred_a_t = self.inv_model(x_comb_t)
            #     inv_loss = F.mse_loss(pred_a_t, a_t)
            # return states[:,:-cond.shape[1]]
            return self.p_losses(x, cond, t, returns)


    

class ARInvModel(nn.Module):
    def __init__(self, hidden_dim, observation_dim, action_dim, low_act=-1.0, up_act=1.0):
        super(ARInvModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.action_embed_hid = 128
        self.out_lin = 128
        self.num_bins = 80

        self.up_act = up_act
        self.low_act = low_act
        self.bin_size = (self.up_act - self.low_act) / self.num_bins
        self.ce_loss = nn.CrossEntropyLoss()

        self.state_embed = nn.Sequential(
            nn.Linear(2 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lin_mod = nn.ModuleList([nn.Linear(i, self.out_lin) for i in range(1, self.action_dim)])
        self.act_mod = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, self.action_embed_hid), nn.ReLU(),
                                                    nn.Linear(self.action_embed_hid, self.num_bins))])

        for _ in range(1, self.action_dim):
            self.act_mod.append(
                nn.Sequential(nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid), nn.ReLU(),
                              nn.Linear(self.action_embed_hid, self.num_bins)))

    def forward(self, comb_state, deterministic=False):
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(self.low_act + l_0 * self.bin_size,
                                              self.low_act + (l_0 + 1) * self.bin_size).sample()

        a = [a_0.unsqueeze(1)]

        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](torch.cat(a, dim=1))], dim=1))
            l_i = torch.distributions.Categorical(logits=lp_i).sample()

            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(self.low_act + l_i * self.bin_size,
                                                  self.low_act + (l_i + 1) * self.bin_size).sample()

            a.append(a_i.unsqueeze(1))

        return torch.cat(a, dim=1)

    def calc_loss(self, comb_state, action):
        eps = 1e-8
        action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
        l_action = torch.div((action - self.low_act), self.bin_size, rounding_mode='floor').long()
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

        for i in range(1, self.action_dim):
            loss += self.ce_loss(self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)),
                                     l_action[:, i])

        return loss/self.action_dim