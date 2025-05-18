import copy
import time

import torch as th
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.vdn import VDNMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num


def calculate_target_q(target_mac, batch, enable_parallel_computing=False, thread_num=4):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)
    with th.no_grad():
        # Set target mac to testing mode
        target_mac.set_evaluation_mode()
        target_mac_out_attention = []
        target_mac_out_diffusion = []
        target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs_attention, target_agent_outs_diffusion, di_loss, di_hh_loss = target_mac.forward(batch, t=t)
            target_mac_out_attention.append(target_agent_outs_attention)
            target_mac_out_diffusion.append(target_agent_outs_diffusion)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out_attention = th.stack(target_mac_out_attention, dim=1)  # Concat across time
        target_mac_out_diffusion = th.stack(target_mac_out_diffusion, dim=1)  # Concat across time
        return target_mac_out_diffusion, target_mac_out_attention

def calculate_n_step_td_target(target_mixer, target_max_qvals, batch, rewards, terminated, mask, gamma, td_lambda,
                               enable_parallel_computing=False, thread_num=4, q_lambda=False, target_mac_out=None):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)

    with th.no_grad():
        # Set target mixing net to testing mode
        target_mixer.eval()
        # Calculate n-step Q-Learning targets
        target_max_qvals = target_mixer(target_max_qvals, batch["state"])

        if q_lambda:
            raise NotImplementedError
            qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
            qvals = target_mixer(qvals, batch["state"])
            targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals, gamma, td_lambda)
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, gamma, td_lambda)
        return targets.detach()


class DITWODECHIDDENLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())
        self.mse = th.nn.MSELoss()

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":  # 31.521K
            self.mixer = Mixer(args)
        else:
            raise "mixer error"

        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0

        self.enable_parallel_computing = (not self.args.use_cuda) and getattr(self.args, 'enable_parallel_computing',
                                                                              False)
        # self.enable_parallel_computing = False
        if self.enable_parallel_computing:
            from multiprocessing import Pool
            # Multiprocessing pool for parallel computing.
            self.pool = Pool(1)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.enable_parallel_computing:
            target_mac_out_diffusion, target_mac_out_attention = self.pool.apply_async(
                calculate_target_q,
                (self.target_mac, batch, True, self.args.thread_num)
            )

        # Calculate estimated Q-Values
        self.mac.set_train_mode()
        mac_out_diffusion = []
        mac_out_attention = []
        di_loss = []
        di_hh_loss = []
        self.mac.init_hidden(batch.batch_size)
        # 训练的时候先atten动作，测试的时候先diffusion动作
        for t in range(batch.max_seq_length):
            agent_outs_diffusion, agent_outs_attention, di_losses, di_hh_losses = self.mac.forward(batch, t=t)
            mac_out_diffusion.append(agent_outs_diffusion)
            mac_out_attention.append(agent_outs_attention)
            di_loss.append(di_losses)
            di_hh_loss.append(di_hh_losses)
        mac_out_diffusion = th.stack(mac_out_diffusion, dim=1)  # Concat over time
        mac_out_attention = th.stack(mac_out_attention, dim=1)  # Concat over time
        di_loss = sum(di_loss)/len(di_loss)
        if self.args.use_hh_loss:
            di_hh_loss = sum(di_hh_loss)/len(di_hh_loss)        
        # TODO: double DQN action, COMMENT: do not need copy
        mac_out_diffusion[avail_actions == 0] = -9999999
        mac_out_attention[avail_actions == 0] = -9999999
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_diffusion = th.gather(mac_out_diffusion[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_attention = th.gather(mac_out_attention[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            if self.enable_parallel_computing:
                target_mac_out_attention = target_mac_out_attention.get()
                target_mac_out_diffusion = target_mac_out_diffusion.get()
            else:
                target_mac_out_diffusion, target_mac_out_attention = calculate_target_q(self.target_mac, batch)
        ############################################################################
        ########################## diffusion部分的训练决策部分 ########################
        ############################################################################
            # Max over target Q-Values/ Double q learning
            # mac_out_detach = mac_out.clone().detach()
            # TODO: COMMENT: do not need copy
            mac_out_detach_diffusion = mac_out_diffusion
            # mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions_diffusion = mac_out_detach_diffusion.max(dim=3, keepdim=True)[1]

            target_max_qvals_diffusion = th.gather(target_mac_out_diffusion, 3, cur_max_actions_diffusion).squeeze(3)

            assert getattr(self.args, 'q_lambda', False) == False
            if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
                targets_diffusion = self.pool.apply_async(
                    calculate_n_step_td_target,
                    (self.target_mixer, target_max_qvals_diffusion, batch, rewards, terminated, mask, self.args.gamma,
                     self.args.td_lambda, True, self.args.thread_num, False, None)
                )
            else:
                targets_diffusion = calculate_n_step_td_target(
                    self.target_mixer, target_max_qvals_diffusion, batch, rewards, terminated, mask, self.args.gamma,
                    self.args.td_lambda
                )

        ############################################################################
        ########################## attention部分的训练决策部分 ########################
        ############################################################################
            # Max over target Q-Values/ Double q learning
            # mac_out_detach = mac_out.clone().detach()
            # TODO: COMMENT: do not need copy
            mac_out_detach_attention = mac_out_attention
            # mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions_attention = mac_out_detach_attention.max(dim=3, keepdim=True)[1]

            target_max_qvals_attention = th.gather(target_mac_out_attention, 3, cur_max_actions_attention).squeeze(3)

            assert getattr(self.args, 'q_lambda', False) == False
            if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
                targets_attention = self.pool.apply_async(
                    calculate_n_step_td_target,
                    (self.target_mixer, target_max_qvals_attention, batch, rewards, terminated, mask, self.args.gamma,
                     self.args.td_lambda, True, self.args.thread_num, False, None)
                )
            else:
                targets_attention = calculate_n_step_td_target(
                    self.target_mixer, target_max_qvals_attention, batch, rewards, terminated, mask, self.args.gamma,
                    self.args.td_lambda
                )

        # Set mixing net to training mode
        self.mixer.train()
        # Mixer
        chosen_action_qvals_diffusion = self.mixer(chosen_action_qvals_diffusion, batch["state"][:, :-1])
        chosen_action_qvals_attention = self.mixer(chosen_action_qvals_attention, batch["state"][:, :-1])

        if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
            targets_diffusion = targets_diffusion.get()
            targets_attention = targets_attention.get()

        td_error_diffusion = (chosen_action_qvals_diffusion - targets_diffusion)
        td_error2_diffusion = 0.5 * td_error_diffusion.pow(2)        
        td_error_attention= (chosen_action_qvals_attention - targets_attention)
        td_error2_attention = 0.5 * td_error_attention.pow(2)

        mask = mask.expand_as(td_error2_diffusion)
        masked_td_error_diffusion = td_error2_diffusion * mask
        masked_td_error_attention = td_error2_attention * mask

        mask_elems = mask.sum()
        # di_loss = self.mse(di_embedding_state, embedding_state)
        loss = masked_td_error_diffusion.sum() / mask_elems + masked_td_error_attention.sum() / mask_elems + di_loss
        if self.args.use_hh_loss:
            loss = loss + self.args.weight_hh_loss * di_hh_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        # print("Avg cost {} seconds".format(self.avg_time))

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # For log
            with th.no_grad():
                mask_elems = mask_elems.item()
                td_error_abs_attention = masked_td_error_attention.abs().sum().item() / mask_elems
                q_taken_mean_attention = (chosen_action_qvals_attention * mask).sum().item() / (mask_elems * self.args.n_agents)
                target_mean_attention = (targets_attention * mask).sum().item() / (mask_elems * self.args.n_agents)                
                td_error_abs_diffusion = masked_td_error_diffusion.abs().sum().item() / mask_elems
                q_taken_mean_diffusion = (chosen_action_qvals_diffusion * mask).sum().item() / (mask_elems * self.args.n_agents)
                target_mean_diffusion = (targets_diffusion * mask).sum().item() / (mask_elems * self.args.n_agents)
            if self.args.use_hh_loss:
                self.logger.log_stat("loss_di_hh", di_hh_loss.item(), t_env)
            self.logger.log_stat("loss_di", di_loss.item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("loss_td_attention", td_error_abs_attention, t_env)
            self.logger.log_stat("q_taken_mean_attention", q_taken_mean_attention, t_env)
            self.logger.log_stat("target_mean_attention", target_mean_attention, t_env)
            self.logger.log_stat("loss_td_diffusion", td_error_abs_diffusion, t_env)
            self.logger.log_stat("q_taken_mean_diffusion", q_taken_mean_diffusion, t_env)
            self.logger.log_stat("target_mean_diffusion", target_mean_diffusion, t_env)

            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def __del__(self):
        if self.enable_parallel_computing:
            self.pool.close()
