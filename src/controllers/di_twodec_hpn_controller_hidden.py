#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch as th

from .basic_controller import BasicMAC


class DataParallelAgent(th.nn.DataParallel):
    def init_hidden(self):
        # make hidden states on same device as model
        return self.module.init_hidden()


# This multi-agent controller shares parameters between agents
class DITWODECHPNMACHIDDEN(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(DITWODECHPNMACHIDDEN, self).__init__(scheme, groups, args)
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1                
        self.di_hidden_states = None
        self.att_hidden_states = None

    # Add new func
    def _get_obs_component_dim(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (6, 5), (4, 5), 1]
        # # 敌人特征中的“j”的个数
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        return (move_feats_dim, enemy_feats_dim_flatten, ally_feats_dim_flatten, own_feats_dim), (
            enemy_feats_dim, ally_feats_dim)

    def _build_inputs(self, batch, t):
        ################    obs_inputs  #####################
        bs = batch.batch_size
        obs_component_dim, _ = self._get_obs_component_dim()
        raw_obs_t = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        # th.split是torch的切分函数，在dim=-1的维度以obs_component_dim元组切割，最终切成：
        # [:, move_feats_dim],[:, enemy_feats_dim_flatten],[:, ally_feats_dim_flatten],[:, own_feats_dim]
        move_feats_t, enemy_feats_t, ally_feats_t, own_feats_t = th.split(raw_obs_t, obs_component_dim, dim=-1)
        enemy_feats_t = enemy_feats_t.reshape(bs * self.n_agents * self.n_enemies,
                                              -1)  # [bs * n_agents * n_enemies, fea_dim]
        ally_feats_t = ally_feats_t.reshape(bs * self.n_agents * self.n_allies,
                                            -1)  # [bs * n_agents * n_allies, a_fea_dim]
        # merge move features and own features to simplify computation.
        context_feats = [move_feats_t, own_feats_t]  # [batch, agent_num, own_dim]
        own_context = th.cat(context_feats, dim=2).reshape(bs * self.n_agents, -1)  # [bs * n_agents, own_dim]

        embedding_indices = []
        if self.args.obs_agent_id:
            # agent-id indices, [bs, n_agents]
            embedding_indices.append(th.arange(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1))
        if self.args.obs_last_action:
            # action-id indices, [bs, n_agents]
            if t == 0:
                embedding_indices.append(None)
            else:
                embedding_indices.append(batch["actions"][:, t - 1].squeeze(-1))
        return bs, own_context, enemy_feats_t, ally_feats_t, embedding_indices
    
    def _get_state_component_dim(self):
        ally_state_feats_dim, enemy_state_feats_dim, ally_state_action_dim = self.args.state_component
        ally_state_feats_flatten = np.prod(ally_state_feats_dim)
        enemy_state_feats_flatten = np.prod(enemy_state_feats_dim)
        ally_state_action_flatten = np.prod(ally_state_action_dim)
        return (ally_state_feats_flatten, enemy_state_feats_flatten, ally_state_action_flatten)
    
    def _build_state_inputs(self, batch, t):
        ################    state_inputs    #####################
        bs = batch.batch_size
        raw_state_t = batch["state"][:, t] # [batch, state_dim]
        state_component_dim = self._get_state_component_dim()
        ally_state_t, enemy_state_t, ally_action_t = th.split(raw_state_t, state_component_dim, dim=-1)
        ally_state_t = ally_state_t.reshape(bs, (self.n_allies+1), -1)  # [bs, (n_allies+1), a_state_dim]
        enemy_state_feats = enemy_state_t.reshape(bs, self.n_enemies, -1)  # [bs, n_enemies, e_state_dim]
        ally_action_t = ally_action_t.reshape(bs, (self.n_allies+1), -1)  # [bs, (n_allies+1), a_action_dim]  
        ally_state_feats = [ally_state_t, ally_action_t]  # [batch, agent_num, own_dim]
        ally_state_feats = th.cat(ally_state_feats, dim=2).reshape(bs , self.n_agents, -1)  # [bs * n_agents, own_dim]
   
        # merge move features and own features to simplify computation.

        return ally_state_feats, enemy_state_feats
        

    def _get_input_shape(self, scheme):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component
        own_context_dim = move_feats_dim + own_feats_dim
        return own_context_dim, enemy_feats_dim, ally_feats_dim
    
    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        state_inputs = self._build_state_inputs(ep_batch, t)
        # state = ep_batch["state"][:, t]

        avail_actions = ep_batch["avail_actions"][:, t]
        if not test_mode:
            agent_outs_diffusion, self.di_hidden_states, agent_outs_attention, self.att_hidden_states, di_loss, di_hh_loss = self.agent(agent_inputs, self.di_hidden_states, state_inputs=state_inputs, att_hidden_state=self.att_hidden_states, test_mode=test_mode)
        else:
            agent_outs_diffusion, self.di_hidden_states, agent_outs_attention, self.att_hidden_states, di_loss, di_hh_loss = self.agent(agent_inputs, self.di_hidden_states, test_mode=test_mode)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs_diffusion = agent_outs_diffusion.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs_attention = agent_outs_attention.reshape(ep_batch.batch_size * self.n_agents, -1)
                
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                
                agent_outs_diffusion[reshaped_avail_actions == 0] = -1e10     
                agent_outs_attention[reshaped_avail_actions == 0] = -1e10

            agent_outs_diffusion = th.nn.functional.softmax(agent_outs_diffusion, dim=-1)
            agent_outs_attention = th.nn.functional.softmax(agent_outs_attention, dim=-1)

        if not test_mode:    
            return agent_outs_diffusion.view(ep_batch.batch_size, self.n_agents, -1), agent_outs_attention.view(ep_batch.batch_size, self.n_agents, -1), di_loss, di_hh_loss 
        else:
            return agent_outs_diffusion.view(ep_batch.batch_size, self.n_agents, -1), agent_outs_attention, di_loss, di_hh_loss 

    def init_hidden(self, batch_size):
        self.di_hidden_states = self.agent.init_hidden()
        self.att_hidden_states = self.agent.init_hidden()
        if self.di_hidden_states is not None:
            self.di_hidden_states = self.di_hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        if self.att_hidden_states is not None:
            self.att_hidden_states = self.att_hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outs_diffusion, agent_outs_attention, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions_diffusion = self.action_selector.select_action(agent_outs_diffusion[bs], avail_actions[bs], t_env, test_mode=test_mode)

        if not test_mode: 
            # 第一种方法RNN，训练输入att的hidden，测试使用di的hidden，
            # 训练的时候用attention的动作，测试的时候用diffusion动作
            # 所以训练的用att的动作做select_actions[0], 执行的时候用diffusion做select_actions[0]
            chosen_actions_attention = self.action_selector.select_action(agent_outs_attention[bs], avail_actions[bs], t_env, test_mode=test_mode)
            if self.args.use_diffusion_episode:
                return chosen_actions_diffusion, chosen_actions_attention
            else:
                return chosen_actions_attention, chosen_actions_diffusion
        else:
            return chosen_actions_diffusion, None

