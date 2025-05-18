import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
import modules.diffuser.utils as utils
from config.locomotion_config import Config
from modules.layer.self_atten import Observe_State_Attention1,Observe_State_Attention2

def kaiming_uniform_(tensor_w, tensor_b, mode='fan_in', gain=12 ** (-0.5)):
    fan = nn.init._calculate_correct_fan(tensor_w.data, mode)
    std = gain / math.sqrt(fan)
    bound_w = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    bound_b = 1 / math.sqrt(fan)
    with th.no_grad():
        tensor_w.data.uniform_(-bound_w, bound_w)
        if tensor_b is not None:
            tensor_b.data.uniform_(-bound_b, bound_b)


class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return th.squeeze(x, dim=1)

# class Diffusion():
    



class HAD_HPN_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(HAD_HPN_Agent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.n_heads = args.hpn_head_num
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.att_dim = args.rnn_hidden_dim // 4
        # self.batch_size_run = args.batch_size_run

        self.obs_att_state = args.obs_att_state
        self.Hypernetworks = args.Hypernetworks
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]  # [n_enemies, feat_dim]
        self.ally_feats_dim = self.ally_feats_dim[-1]  # [n_allies, feat_dim]
        self.new_columns = max(self.own_feats_dim,self.ally_feats_dim)
        self.obs_dim = self.own_feats_dim+self.ally_feats_dim*self.n_allies+self.n_enemies*self.enemy_feats_dim
        if self.args.model == 'models.Transformer':
            self.model = utils.Config(
                # Config.model,
                self.args.model,
                savepath=None,
                horizon=Config.horizon,
                #diffuision中，model的输出
                # transition_dim=self.args.state_shape + self.args.obs_shape + self.args.rnn_hidden_dim,
                # transition_dim=self.args.state_shape,
                transition_dim=self.n_agents*self.new_columns + self.n_enemies*self.enemy_feats_dim,
                #diffuision中，model的输入
                cond_dim=self.args.obs_shape + self.args.rnn_hidden_dim,
                dim_mults=Config.dim_mults,
                returns_condition=Config.returns_condition,
                dim=Config.dim,
                condition_dropout=Config.condition_dropout,
                calc_energy=Config.calc_energy,
                device=Config.device,
                mlp_node=self.args.mlp_mode
                # n_position=self.batch_size_run*self.n_agents,
            )()
        else:
            self.model = utils.Config(
                # Config.model,
                self.args.model,
                savepath=None,
                horizon=Config.horizon,
                #diffuision中，model的输出
                # transition_dim=self.args.state_shape + self.args.obs_shape + self.args.rnn_hidden_dim,
                # transition_dim=self.args.state_shape,
                transition_dim=self.n_agents*self.new_columns + self.n_enemies*self.enemy_feats_dim,
                #diffuision中，model的输入
                cond_dim=self.args.obs_shape + self.args.rnn_hidden_dim,
                dim_mults=Config.dim_mults,
                returns_condition=Config.returns_condition,
                dim=Config.dim,
                condition_dropout=Config.condition_dropout,
                calc_energy=Config.calc_energy,
                device=Config.device,
                # n_position=self.batch_size_run*self.n_agents,
            )()

        self.diffusion = utils.Config(
            Config.diffusion,
            savepath=None,
            horizon=Config.horizon,
            #diffuision中，总的输入
            observation_dim=self.args.obs_shape + self.args.rnn_hidden_dim,
            # 输出是状态的形状
            # action_dim=args.rnn_hidden_dim,
            # action_dim=self.args.state_shape,
            action_dim=self.n_agents*self.new_columns + self.n_enemies*self.enemy_feats_dim,
            n_timesteps=self.args.n_diffusion_steps,
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            # predict_epsilon=Config.predict_epsilon,
            predict_epsilon=self.args.predict_epsilon,
            hidden_dim=Config.hidden_dim,
            ar_inv=Config.ar_inv,
            train_only_inv=Config.train_only_inv,
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            condition_guidance_w=Config.condition_guidance_w,
            device=Config.device,
        )(self.model)
        # [4 + 1, (6, 5), (4, 5)]

        ########################################################################################
        #################################  attention过程  ##############################
        ########################################################################################
        if self.args.attention_mode == 1:
            self.att_ally = Observe_State_Attention1(self.new_columns, 
                                                (self.args.state_ally_feats_size + self.n_actions), 
                                                args.att_heads, args.att_embed_dim)
            self.att_enemy = Observe_State_Attention1(self.enemy_feats_dim, 
                                                self.args.state_enemy_feats_size,
                                                args.att_heads, args.att_embed_dim)
        elif self.args.attention_mode == 2:
            self.att_ally = Observe_State_Attention2(self.new_columns, 
                                                (self.args.state_ally_feats_size + self.n_actions), 
                                                args.att_heads, args.att_embed_dim)
            self.att_enemy = Observe_State_Attention2(self.enemy_feats_dim, 
                                                self.args.state_enemy_feats_size,
                                                args.att_heads, args.att_embed_dim)
        self.fc_att_ally = nn.Linear(args.att_embed_dim, self.new_columns)


        self.fc_att_enemy = nn.Linear(args.att_embed_dim, self.enemy_feats_dim)      

        if self.obs_att_state:
            if self.args.map_type == "MMM":
                assert self.n_enemies >= self.n_agents, "For MMM map, for the reason that the 'attack' and 'rescue' use the same ids in SMAC, n_enemies must >= n_agents"
                self.att_hyper_ally = nn.Sequential(
                    nn.Linear(self.new_columns*2, args.hpn_hyper_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.hpn_hyper_dim, ((self.new_columns*2 + 1) * self.rnn_hidden_dim + 1) * self.n_heads)
                )  # output shape: (state_ally_feats_size + self.n_actions) * rnn_hidden_dim + rnn_hidden_dim + 1, for 'rescue actions'
                self.unify_output_heads_rescue = Merger(self.n_heads, 1)
            else:
                self.att_hyper_ally = nn.Sequential(
                    nn.Linear(self.new_columns*2, args.hpn_hyper_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.hpn_hyper_dim, (self.new_columns*2) * self.rnn_hidden_dim * self.n_heads)
                )  # output shape: (state_ally_feats_size + self.n_actions) * rnn_hidden_dim * self.n_heads

            # Multiple entities (use hyper net to process these features to ensure permutation invariant)
            self.att_hyper_enemy = nn.Sequential(
                nn.Linear(self.enemy_feats_dim*2, args.hpn_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.hpn_hyper_dim, ((self.enemy_feats_dim*2 + 1) * self.rnn_hidden_dim + 1) * self.n_heads)
            )  # output shape: (state_enemy_feats_size * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads
            self.unify_att_heads = Merger(self.n_heads, self.rnn_hidden_dim)

        else:
            if self.args.map_type == "MMM":
                assert self.n_enemies >= self.n_agents, "For MMM map, for the reason that the 'attack' and 'rescue' use the same ids in SMAC, n_enemies must >= n_agents"
                self.att_hyper_ally = nn.Sequential(
                    nn.Linear(self.new_columns, args.hpn_hyper_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.hpn_hyper_dim, ((self.new_columns + 1) * self.rnn_hidden_dim + 1) * self.n_heads)
                )  # output shape: (state_ally_feats_size + self.n_actions) * rnn_hidden_dim + rnn_hidden_dim + 1, for 'rescue actions'
                self.unify_output_heads_rescue = Merger(self.n_heads, 1)
            else:
                self.att_hyper_ally = nn.Sequential(
                    nn.Linear(self.new_columns, args.hpn_hyper_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.hpn_hyper_dim, (self.new_columns) * self.rnn_hidden_dim * self.n_heads)
                )  # output shape: (state_ally_feats_size + self.n_actions) * rnn_hidden_dim * self.n_heads

            # Multiple entities (use hyper net to process these features to ensure permutation invariant)
            self.att_hyper_enemy = nn.Sequential(
                nn.Linear(self.enemy_feats_dim, args.hpn_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.hpn_hyper_dim, ((self.enemy_feats_dim + 1) * self.rnn_hidden_dim + 1) * self.n_heads)
            )  # output shape: (state_enemy_feats_size * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads
            self.unify_att_heads = Merger(self.n_heads, self.rnn_hidden_dim)
            


        if self.args.obs_agent_id:
            # embedding table for agent_id
            # Embedding将输入的整数序列转换为密集向量表示,将每个agents的id表示成一个向量，从而方便进行下一步的计算和处理
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, self.rnn_hidden_dim)

        # if self.args.obs_last_action :
            # embedding table for action id
            # 将每个动作的id表示成一个向量，从而方便进行下一步的计算和处理
        self.action_id_embedding = th.nn.Embedding(self.n_actions, self.rnn_hidden_dim)

        # Unique Features (do not need hyper net，不需要超网络)
        self.fc1_own = nn.Linear(self.own_feats_dim, self.rnn_hidden_dim, bias=True)  # only one bias is OK

        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2_normal_actions = nn.Linear(self.rnn_hidden_dim, args.output_normal_actions)  # (no_op, stop, up, down, right, left)
        self.unify_output_heads = Merger(self.n_heads, 1)
        # Reset parameters for hypernets
        # self._reset_hypernet_parameters(init_type="xavier")
        # self._reset_hypernet_parameters(init_type="kaiming")
        self.loss_MSE = th.nn.MSELoss()

    def _reset_hypernet_parameters(self, init_type='kaiming'):
        gain = 2 ** (-0.5)
        # %%%%%%%%%%%%%%%%%%%%%% Hypernet-based API input layer %%%%%%%%%%%%%%%%%%%%
        for m in self.hyper_enemy.modules():
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    kaiming_uniform_(m.weight, m.bias, gain=gain)
                else:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.)
        for m in self.hyper_ally.modules():
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    kaiming_uniform_(m.weight, m.bias, gain=gain)
                else:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1_own.weight.new(1, self.rnn_hidden_dim).zero_()
    
    def pi(self, bs, own_feats_t, embedding_indices, att_ally_feats, att_enemy_feats):
        ########################################################################################
        #######################################  PI过程  #######################################
        ########################################################################################
        # (1) Own feature，自特征，不需要超网络，输出维度rnn_hidden_dim即64
        embedding_own = self.fc1_own(own_feats_t)  # [bs * n_agents, rnn_hidden_dim]

        # (2) ID embeddings
        # Embedding将输入的整数序列转换为密集向量表示,将每个agents的id表示成一个向量，从而方便进行下一步的计算和处理
        if self.args.obs_agent_id:
        # [bs , n_agents]
            agent_indices = embedding_indices[0]
        # [bs * n_agents, rnn_hidden_dim]
            embedding_own = embedding_own + self.agent_id_embedding(agent_indices).view(-1, self.rnn_hidden_dim)
        if self.args.obs_last_action:
            last_action_indices = embedding_indices[-1]
            if last_action_indices is not None:  # t != 0
            # [bs * n_agents, rnn_hidden_dim]
                embedding_own = embedding_own + self.action_id_embedding(last_action_indices).view(
                    -1, self.rnn_hidden_dim)

        # (2) Ally att features
        # [bs*n_agents, ally_feats_dim * rnn_hidden_dim * n_heads]
        att_hyper_ally_out = self.att_hyper_ally(att_ally_feats)
        if self.obs_att_state:
            if self.args.map_type == "MMM":
            # [bs * n_agents, ally_fea_dim, rnn_hidden_dim * head]
                fc1_att_w_ally = att_hyper_ally_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
                    -1, self.new_columns*2, self.rnn_hidden_dim * self.n_heads)
            else:
            # [bs * n_agents, ally_fea_dim, rnn_hidden_dim * head]
                fc1_att_w_ally = att_hyper_ally_out.view(-1, self.new_columns*2, self.rnn_hidden_dim * self.n_heads)
            # [bs * n_agents, 1, ally_fea_dim + self.n_actions] * [bs * n_agents, ally_fea_dim + self.n_actions, n_heads* rnn_hidden_dim] = [bs * n_agents, 1, n_heads*rnn_hidden_dim]
            embedding_att_allies = th.matmul(att_ally_feats.unsqueeze(1), fc1_att_w_ally).view(
                bs * self.n_agents, self.n_agents, self.n_heads, self.rnn_hidden_dim)  # [bs * n_agents, n_agents, head, rnn_hidden_dim]
            embedding_att_allies = embedding_att_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, head, rnn_hidden_dim]

        # (3) Enemy att feature  (enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads
        # enemy_feats_t:[bs*n_agents*n_enemies,e_fea_dim]
            att_hyper_enemy_out = self.att_hyper_enemy(att_enemy_feats)#(enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads
            # [bs * n_agents * n_enemies, state_enemy_feats_size, rnn_hidden_dim * n_heads]
            fc1_att_w_enemy = att_hyper_enemy_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
                    -1, self.enemy_feats_dim*2, self.rnn_hidden_dim * self.n_heads)
    
        else:
            if self.args.map_type == "MMM":
            # [bs * n_agents, ally_fea_dim, rnn_hidden_dim * head]
                fc1_att_w_ally = att_hyper_ally_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
                    -1, self.new_columns, self.rnn_hidden_dim * self.n_heads)
            else:
            # [bs * n_agents, ally_fea_dim, rnn_hidden_dim * head]
                fc1_att_w_ally = att_hyper_ally_out.view(-1, self.new_columns, self.rnn_hidden_dim * self.n_heads)
            # [bs * n_agents, 1, ally_fea_dim + self.n_actions] * [bs * n_agents, ally_fea_dim + self.n_actions, n_heads* rnn_hidden_dim] = [bs * n_agents, 1, n_heads*rnn_hidden_dim]
            embedding_att_allies = th.matmul(att_ally_feats.unsqueeze(1), fc1_att_w_ally).view(
                bs * self.n_agents, self.n_agents, self.n_heads, self.rnn_hidden_dim)  # [bs * n_agents, n_agents, head, rnn_hidden_dim]
            embedding_att_allies = embedding_att_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, head, rnn_hidden_dim]

        # (3) Enemy att feature  (enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads
        # enemy_feats_t:[bs*n_agents*n_enemies,e_fea_dim]
            att_hyper_enemy_out = self.att_hyper_enemy(att_enemy_feats)#(enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads
            # [bs * n_agents * n_enemies, state_enemy_feats_size, rnn_hidden_dim * n_heads]
            fc1_att_w_enemy = att_hyper_enemy_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
                    -1, self.enemy_feats_dim, self.rnn_hidden_dim * self.n_heads)
            # torch.matmul两个张量矩阵相乘
        # [bs * n_agents * n_enemies, 1, state_enemy_feats_size] * [bs * n_agents * n_enemies, state_enemy_feats_size, rnn_hidden_dim * n_heads] = [bs * n_agents * n_enemies, 1, rnn_hidden_dim]
        embedding_att_enemies = th.matmul(att_enemy_feats.unsqueeze(1), fc1_att_w_enemy).view(
            bs * self.n_agents, self.n_enemies, self.n_heads, self.rnn_hidden_dim)  # [bs * n_agents, n_enemies, n_heads, rnn_hidden_dim]
        # sum(dim=1)按照维度1加和，加起来。
        # keepdim就似乎size里面是否还保留这个压缩起来的（因为注定是1了，其实没有信息量）
        embedding_att_enemies = embedding_att_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, rnn_hidden_dim]

        # Final embedding
        embedding_att =self.unify_att_heads(embedding_att_allies + embedding_att_enemies)  # [bs * n_agents, head, rnn_hidden_dim]
        # embedding_state:[bs * n_agent, rnn_hidden_dim], y:[bs * n_agent, rnn_hidden_dim * 2]
        
    # Final embedding
        final_embedding = embedding_own + embedding_att

    ########################################################################################
    ###################################  模块B与C  #################################
    ########################################################################################
    # 把di_embedding_state整合进embedding中作为输入
        x = F.relu(final_embedding, inplace=True)
    ###########################
        return x, att_hyper_ally_out, att_hyper_enemy_out


    def rnn_net(self, x, h_in):
        hh = self.rnn(x, h_in)  # [bs * n_agents, rnn_hidden_dim]
        return hh

    def pe(self, bs, hh, att_hyper_ally_out, att_hyper_enemy_out):
    # Q-values of normal actions
        q_normal = self.fc2_normal_actions(hh).view(bs, self.n_agents, -1)  # [bs, n_agents, 6]

    ########################################################################################
    ########################################  PE过程  ######################################
    ########################################################################################

    # Q-values of attack actions: [bs * n_agents * n_enemies, rnn_hidden_dim * n_heads]
        fc2_w_attack = att_hyper_enemy_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads: -self.n_heads].reshape(
            bs * self.n_agents, self.n_enemies, self.rnn_hidden_dim, self.n_heads
        ).transpose(1, 2).reshape(  # -> [bs * n_agents, rnn_hidden_dim, n_enemies, n_heads]
            bs * self.n_agents, self.rnn_hidden_dim, self.n_enemies * self.n_heads
        )  # [bs * n_agents, rnn_hidden_dim, n_enemies * heads]
        fc2_b_attack = att_hyper_enemy_out[:, -self.n_heads:].reshape(bs * self.n_agents, self.n_enemies * self.n_heads)

    # [bs*n_agents, 1, rnn_hidden_dim] * [bs*n_agents, rnn_hidden_dim, n_enemies*head] -> [bs*n_agents, 1, n_enemies*head]
        q_attacks = (th.matmul(hh.unsqueeze(1), fc2_w_attack).squeeze(1) + fc2_b_attack).view(
            bs * self.n_agents * self.n_enemies, self.n_heads, 1
        )  # [bs * n_agents, n_enemies*head] -> [bs * n_agents * n_enemies, head, 1]

    # Merge multiple heads into one.
        q_attack = self.unify_output_heads(q_attacks).view(  # [bs * n_agents * n_enemies, 1]
            bs, self.n_agents, self.n_enemies
        )  # [bs, n_agents, n_enemies]

    # %%%%%%%%%%%%%%% 'rescue' actions for map_type == "MMM" %%%%%%%%%%%%%%%
        if self.args.map_type == "MMM":
            fc2_w_rescue = att_hyper_ally_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads: -self.n_heads].reshape(
                bs * self.n_agents, self.n_agents, self.rnn_hidden_dim, self.n_heads
            ).transpose(1, 2).reshape(  # -> [bs * n_agents, rnn_hidden_dim, n_allies, n_heads]
                bs * self.n_agents, self.rnn_hidden_dim, self.n_agents * self.n_heads
            )  # [bs * n_agents, rnn_hidden_dim, n_allies * heads]
            fc2_b_rescue = att_hyper_ally_out[:, -self.n_heads:].reshape(bs * self.n_agents, self.n_agents * self.n_heads)
        # [bs*n_agents, 1, rnn_hidden_dim] * [bs*n_agents, rnn_hidden_dim, n_allies*head] -> [bs*n_agents, 1, n_allies*head]
            q_rescues = (th.matmul(hh.unsqueeze(1), fc2_w_rescue).squeeze(1) + fc2_b_rescue).view(
                bs * self.n_agents * self.n_agents, self.n_heads, 1
            )  # [bs * n_agents, n_allies*head] -> [bs * n_agents * n_allies, head, 1]
        # Merge multiple heads into one.
            q_rescue = self.unify_output_heads_rescue(q_rescues).view(  # [bs * n_agents * n_allies, 1]
                bs, self.n_agents, self.n_agents
            )  # [bs, n_agents, n_allies]

        # For the reason that medivac is the last indexed agent, so the rescue action idx -> [0, n_allies-1]
            right_padding = th.ones_like(q_attack[:, -1:, self.n_agents:], requires_grad=False) * (-9999999)
            modified_q_attack_of_medivac = th.cat([q_rescue[:, -1:, :], right_padding], dim=-1)
            q_attack = th.cat([q_attack[:, :-1], modified_q_attack_of_medivac], dim=1)

    # Concat 2 types of Q-values
        q = th.cat((q_normal, q_attack), dim=-1)  # [bs, n_agents, 6 + n_enemies]

        return q.view(bs, self.n_agents, -1)
   
 



    def forward(self, inputs, di_hidden_state, state_inputs=None, att_hidden_state=None, test_mode=False):
        # [bs*n_agents,own_dim],[bs*n_agents*n_enemies,e_fea_dim],[bs*n_agents*n_allies,a_fea_dim]
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs
        di_h_in = di_hidden_state.reshape(-1, self.rnn_hidden_dim)
        di_ally_feats_t = ally_feats_t.reshape(bs*self.n_agents, -1)
        di_enemy_feats_t = enemy_feats_t.reshape(bs*self.n_agents, -1)     
        all_obs = th.cat((own_feats_t, di_ally_feats_t, di_enemy_feats_t), dim=-1)


        # (1) 生成atten要用的观测信息
        ally_feats = ally_feats_t.cpu().numpy()
        ally_feats = np.pad(ally_feats, ((0, 0), (0, self.new_columns - ally_feats.shape[1])), mode='constant')
        own_feats = own_feats_t.cpu().numpy()

        for i in range(own_feats_t.shape[0]):
            ally_feats = np.insert(ally_feats, i * self.n_agents + i % self.n_agents, own_feats[i], axis=0)
        ally_feats = th.from_numpy(ally_feats).to(own_feats_t.device)

        if not test_mode:
        ########################################################################################
        #############  训练时attention生成information，并以此为groundtruth训练diffusion  ###########
        ######################################################################################## 


            # (2) 生成atten要用的真实状态信息
            ally_state_t, enemy_state_t = state_inputs       
            ally_state_feats=ally_state_t.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
            enemy_state_feats=enemy_state_t.unsqueeze(1).repeat(1, self.n_agents, 1, 1)               
            ally_state_feats=ally_state_feats.reshape(bs*self.n_agents, self.n_agents, (self.args.state_ally_feats_size + self.n_actions))
            enemy_state_feats=enemy_state_feats.reshape(bs*self.n_agents, self.n_enemies, self.args.state_enemy_feats_size)
            att_h_in = att_hidden_state.reshape(-1, self.rnn_hidden_dim)
                
            # (3)进行训练过程的attention
            att_ally_feats = self.att_ally(ally_feats.view(bs * self.n_agents, self.n_agents, -1), 
                                       ally_state_feats)
            att_ally_feats = self.fc_att_ally(att_ally_feats).reshape(bs * self.n_agents, -1)

            att_enemy_feats = self.att_enemy(enemy_feats_t.view(bs * self.n_agents, self.n_enemies, -1), 
                                         enemy_state_feats)
            att_enemy_feats = self.fc_att_enemy(att_enemy_feats).reshape(bs * self.n_agents, -1)

            # (4)训练时用attention的information做groundtruth,观测与隐藏层做输入
            di_loss = self.diffusion.loss(th.cat((att_ally_feats,att_enemy_feats),dim=-1).detach(), th.cat((all_obs, di_h_in), dim=-1))

            if self.args.ues_KL == False:
                di_loss = di_loss
            else:
                di_loss = di_loss + F.kl_div(self.diffusion.forward(th.cat((all_obs, di_h_in), dim=-1)).softmax(-1).log(), th.cat((att_ally_feats,att_enemy_feats),dim=-1).softmax(-1).detach(), reduction='sum')


        ########################################################################################
        ############################  执行过程diffusion预测information  ##########################
        ######################################################################################## 

            # concat
            if self.obs_att_state == True:
                # att_ally_feats = self.fc_all_ally(ally_feats) + att_ally_feats
                att_ally_feats = th.cat([ally_feats.reshape(bs * self.n_agents * self.n_agents, -1),att_ally_feats.reshape(bs * self.n_agents * self.n_agents, -1)], dim=-1)
                att_enemy_feats = th.cat([enemy_feats_t.reshape(bs * self.n_agents * self.n_enemies, -1),att_enemy_feats.reshape(bs * self.n_agents * self.n_enemies, -1)], dim=-1)
                # att_ally_feats = ally_feats + att_ally_feats
            else :
                att_ally_feats = att_ally_feats.reshape(bs * self.n_agents * self.n_agents, -1)
                att_enemy_feats = att_enemy_feats.reshape(bs * self.n_agents * self.n_enemies, -1)
        

            x, att_hyper_ally_out, att_hyper_enemy_out = self.pi(bs, own_feats_t, embedding_indices, att_ally_feats, att_enemy_feats)
            att_hh = self.rnn_net(x, att_h_in)
            q_att = self.pe(bs, att_hh, att_hyper_ally_out, att_hyper_enemy_out)


        attention_information=self.diffusion.forward(th.cat((all_obs, di_h_in), dim=-1))
        di_att_ally_feats = attention_information[:, :(self.n_agents * self.new_columns)]
        di_att_enemy_feats = attention_information[:, -(self.n_enemies * self.enemy_feats_dim):]
        if self.obs_att_state == True:
        # att_ally_feats = self.fc_all_ally(ally_feats) + att_ally_feats
            di_att_ally_feats = th.cat([ally_feats.reshape(bs * self.n_agents * self.n_agents, -1),di_att_ally_feats.reshape(bs * self.n_agents * self.n_agents, -1)], dim=-1)
            di_att_enemy_feats = th.cat([enemy_feats_t.reshape(bs * self.n_agents * self.n_enemies, -1),di_att_enemy_feats.reshape(bs * self.n_agents * self.n_enemies, -1)], dim=-1)
        # att_ally_feats = ally_feats + att_ally_feats
        else :
            di_att_ally_feats = di_att_ally_feats.reshape(bs * self.n_agents * self.n_agents, -1)
            di_att_enemy_feats = di_att_enemy_feats.reshape(bs * self.n_agents * self.n_enemies, -1)

        di_x, di_att_hyper_ally_out, di_att_hyper_enemy_out = self.pi(bs, own_feats_t, embedding_indices, di_att_ally_feats, di_att_enemy_feats)
        di_hh = self.rnn_net(di_x, di_h_in)
        q_diffusion = self.pe(bs, di_hh, di_att_hyper_ally_out, di_att_hyper_enemy_out)

        # if not test_mode:
        #     if self.args.use_hh_loss:
        #         di_loss = di_loss + 
        # use_attention_hh, 
        # 如果各自用各自的，那输入就得多一个hh做hidden，而且di_hh还得作为diffusion输入,也就是输入俩返回俩
        if not test_mode:
            if self.args.use_hh_loss:
                return q_diffusion, di_hh.view(bs, self.n_agents, -1), q_att, att_hh.view(bs, self.n_agents, -1), di_loss, self.loss_MSE(di_hh, att_hh.detach())
            else:
                return q_diffusion, di_hh.view(bs, self.n_agents, -1), q_att, att_hh.view(bs, self.n_agents, -1), di_loss, None
        else:
            return q_diffusion, di_hh.view(bs, self.n_agents, -1), None, None, None, None
        
        # if not test_mode:
        #     return q_diffusion, hh.view(bs, self.n_agents, -1), di_loss
        # else:
        #     return q_diffusion, di_hh.view(bs, self.n_agents, -1), None


