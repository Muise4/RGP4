import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import sys

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

# run.py 文件中run函数的主要作用是构建实验参数变量 args 以及一个自定义 Logger 类的记录器 logger
def run(_run, _config, _log):
    # run是run函数类<sacred.run.Run>
    # config是之前的参数配置
    # log是<Logger my_main (DEBUG)>
    # 内置变量_config的拷贝作为参数传入到了run函数中，_config 是字典变量，因此查看参数时，需要利用 _config[key]=value，
    # 在 run 函数中，构建了一个namespace类的变量args，将_config中的参数都传给了 args，
    # 这样就可以通过args.key=value的方式查看参数了。

    # check args sanity
    _config = args_sanity_check(_config, _log)
    # SN函数是sys里面的一个配置
    # 通过访问sys.implementation属性，你可以获得当前正在运行的Python解释器的实现信息。这些信息对于了解解释器的特性、进行调试、日志记录、版本检查等任务非常有用。
    # # 访问sys.implementation属性的不同属性  
    # print(sys.implementation.name)        # 输出解释器的名称，例如"cpython"  
    # print(sys.implementation.version)     # 输出解释器的版本号，例如(3, 8, 5)  
    # print(sys.implementation.hexversion)  # 输出解释器的版本号，以十六进制整数表示，例如0x30805  
    # print(sys.implementation.cache_tag)   # 输出用于缓存编译的Python代码的标签
    args = SN(**_config)

    th.set_num_threads(args.thread_num)
    # th.set_num_interop_threads(8)

    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    # _log就是一个记录的,并不会参与整合_config
    # pymarl自定义了一个utils.logging.Logger类的对象logger对ex的内置变量_run和_log进行封装，
    # 最终所有的实验结果通过 logger.log_stat(key, value, t, to_sacred=True) 记录在了./results/sacred/实验编号/info.json文件中。
    # 在整个实验中，logger主要对runner和learner两个对象所产生的实验数据进行了记录，包括训练数据和测试数据的如下内容：
    # runner对象：
        # 一系列环境特定的实验数据，即env_info，在SC2中，包括"battle_won_mean"，"dead_allies_mean"， "dead_enemies_mean";
        # 训练相关的实验数据：包括"ep_length_mean"，"epsilon"，"return_mean"；
    # learner对象：
        # 训练相关的实验数据："loss"，"grad_norm"，"td_error_abs" ，"q_taken_mean"，"target_mean"。
    logger = Logger(_log)
    # 输出经过格式化的参数,等级为info
    _log.info("Experiment Parameters:")
    # pformat是调整输出格式的,嵌套会自动缩进
    # indent：指定缩进的空格数，默认为1。缩进用于表示嵌套对象的层次结构。
    # width：指定输出的最大宽度，默认为80。如果输出的字符串超过了这个宽度，会自动进行换行。
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")就是当前时间,后面的%是年月日时分秒
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

    testing_algorithms = ["vdn", "qmix", "hpn_vdn", "hpn_qmix", 
                          "di_hpn_qmix", "di_PE_hpn_qmix", "di_PEN_hpn_qmix", 
                          "di_PE_con_hpn_qmix", "di_PE_com_hpn_qmix", 
                          "di_PE_att_hpn_qmix", "di_PE_com_att_hpn_qmix", 
                          "di_att_hpn_qmix", "di_att_com_hpn_qmix", "di_attgai_hpn_qmix_state",
                          "di_attgai_hpn_qmix", "di_HE_hpn_qmix", "di_hp_hpn_qmix",
                          "qmix_cadp", "qplex_cadp", "vdn_cadp", "di_attgai_hpn_vdn",
                          "di_astate_hpn_qmix_state","di_MLPattgai_hpns_rnn_state","di_allobsattgai_hpn_qmix_state",
                          "deepset_vdn", "deepset_qmix", "deepset_hyper_vdn", "deepset_hyper_qmix",
                          "updet_vdn", "updet_qmix", "vdn_DA", "qmix_DA",
                          "gnn_vdn", "gnn_qmix", "qplex", "hpn_qplex", "asn",
                          "PTDE_rnn_agent_state", "di_atttogai_hpn_qmix", "di_atttogai_hpn_qmix_state",
                          "di_atttogai_hpn_qmix_stage", "di_attto_hpn_qmix", "di_atttogai_hpn_qmix_stage_notd",
                          "di_atttogai_hpn_qmix_twodec", "di_atttogai_hpn_qmix_twodec_hidden", 
                          "di_atttogai_hpn_qmix_twodec_hidden_state_z",
                          "di_atttogai_hpn_qmix_twodec_hidden_target", "di_atttogai_hpn_twornn_twodec_hidden"
                          "had_hpn_qmix", "had_hpn_qmix_nohis", "had_qmix", "had_qmix_noder", "had_qmix_nohis", 
                          "had_vdn", "had_vdn_noder", "had_vdn_nohis","had_hpn_vdn","sidiff_qmix",
                          "had_qmix_MLP", "had_qmix_GAN", "had_qmix_VAE", 
                          "had_qmix_MLP_global_state", "had_qmix_MLP_nodiloss", "had_qmix_MLP_nohhloss", "had_qmix_MLP_notdloss", 
                          "had_qmix_global_state", "had_qmix_nodiloss", "had_qmix_nohhloss", "had_qmix_notdloss", 
                          "local_state_sidiff"
                          ]
    env_name = args.env
    # 直到138行全是在写logdir
    logdir = env_name
    if env_name in ["sc2", "sc2_v2", ]:
        logdir = os.path.join("{}_{}".format(
            logdir,
            args.env_args["map_name"],
        ))
        # logdir = os.path.join("{}_{}-obs_aid={}-obs_act={}".format(
        #     logdir,
        #     args.env_args["map_name"],
        #     int(args.obs_agent_id),
        #     int(args.obs_last_action),
        # ))
        # if env_name == "sc2_v2":
        #     logdir = logdir + "-conic_fov={}".format(
        #         "1-change_fov_by_move={}".format(
        #             int(args.env_args["change_fov_with_move"])) if args.env_args["conic_fov"] else "0"
        #     )
        # if env_name == "sc2_v2":
        #     logdir = logdir + "-{}".format(args.env_args["angle"])
    # logdir = os.path.join(logdir,
    #                       "algo={}-agent={}".format(args.name, args.agent),
    #                       "env_n={}".format(
    #                           args.batch_size_run,
    #                       ))
    if args.name in testing_algorithms:
        if args.name in ["vdn_DA", "qmix_DA", ]:
            logdir = os.path.join(logdir,
                                  "{}-data_augment={}".format(
                                      args.mixer, args.augment_times
                                  ))
        elif args.name in ["gnn_vdn", "gnn_qmix"]:
            logdir = os.path.join(logdir,
                                  "{}-layer_num={}".format(
                                      args.mixer, args.gnn_layer_num
                                  ))
        elif args.name in ["vdn", "qmix", "deepset_vdn", "deepset_qmix", "qplex", "asn"]:
            logdir = os.path.join(logdir,
                                  "mixer={}".format(
                                      args.mixer,
                                  ))
        elif args.name in ["updet_vdn", "updet_qmix"]:
            logdir = os.path.join(logdir,
                                  "mixer={}-att_dim={}-att_head={}-att_layer={}".format(
                                      args.mixer,
                                      args.transformer_embed_dim,
                                      args.transformer_heads,
                                      args.transformer_depth,
                                  ))
        elif args.name in ["deepset_hyper_vdn", "deepset_hyper_qmix"]:
            logdir = os.path.join(logdir,
                                  "mixer={}-hpn_hyperdim={}".format(
                                      args.mixer,
                                      args.hpn_hyper_dim,
                                  ))
        elif args.name in ["hpn_vdn", "hpn_qmix", "had_hpn_qmix", "hpn_qplex"]:
            logdir = os.path.join(logdir,
                                  "head_n={}-mixer={}-hpn_hyperdim={}-acti={}".format(
                                      args.hpn_head_num,
                                      args.mixer,
                                      args.hpn_hyper_dim,
                                      args.hpn_hyper_activation,
                                  ))

    # logdir = os.path.join(logdir,
    #                       "rnn_dim={}-2bs={}_{}-tdlambda={}-epdec_{}={}k".format(
    #                           args.rnn_hidden_dim,
    #                           args.buffer_size,
    #                           args.batch_size,
    #                           args.td_lambda,
    #                           args.epsilon_finish,
    #                           args.epsilon_anneal_time // 1000,
    #                       ))
    args.log_model_dir = logdir
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), args.local_results_path, "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        if args.name in testing_algorithms:  # add parameter config to the logger path！
            tb_exp_direc = os.path.join(tb_logs_direc, logdir, unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    # 评估序列
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # run_sequential 是实验运行的主要函数，作用是首先是构建如下自定义类的对象：
    # EpisodeRunner类的环境运行器对象runner ，“runner是环境运行器”
    # ReplayBuffer类的经验回放池对象buffer，
    # BasicMAC类的智能体控制器对象mac，
    # QLearner类的智能体学习器对象learner，
    # 最后进行实验，即训练智能体，记录实验结果，定期测试并保存模型。

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    
    # 把环境的一些信息给args。
    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    if args.env in ["sc2", "sc2_v2", "gfootball"]:
        if args.env in ["sc2", "sc2_v2"]:
            args.output_normal_actions = env_info["n_normal_actions"]
        args.n_enemies = env_info["n_enemies"]
        args.n_allies = env_info["n_allies"]
        # args.obs_ally_feats_size = env_info["obs_ally_feats_size"]
        # args.obs_enemy_feats_size = env_info["obs_enemy_feats_size"]
        args.state_ally_feats_size = env_info["state_ally_feats_size"]
        args.state_enemy_feats_size = env_info["state_enemy_feats_size"]
        args.obs_component = env_info["obs_component"]
        args.state_component = env_info["state_component"]
        args.map_type = env_info["map_type"]
        args.agent_own_state_size = env_info["state_ally_feats_size"]
        args.unit_dim = env_info["state_ally_feats_size"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    # [batch, episode_length, n_agents, feature_dim]

    # buffer对象属于自定义的components.episode_buffer.ReplayBuffer(EpisodeBatch)类，该对象的主要作用是存储样本以及采样样本。
    # ReplayBuffer的父类是EpisodeBatch。EpisodeBatch类对象用于存储episode的样本，
        #  ReplayBuffer(EpisodeBatch)类对象则用于存储所有的off-policy样本，
        # 也即EpisodeBatch类变量的样本会持续地补充到ReplayBuffer(EpisodeBatch)类的变量中。
    # 同样由于QMix用的是DRQN结构，因此EpisodeBatch与ReplayBuffer中的样本都是以episode为单位存储的。
    # 在EpisodeBatch中数据的维度是[batch_size, max_seq_length, *shape]，
    # ‘‘EpisodeBatch中Batch Size表示此时batch中有多少episode，’’
    # ReplayBuffer类数据的维度是[buffer_size, max_seq_length, *shape]。
    # ReplayBuffer中episodes_in_buffer表示此时buffer中有多少个episode的有效样本。
    # max_seq_length则表示一个episode的最大长度。
    # buffer经验回放池
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    # mac对象中的一个重要属性就是nn.module类的智能体对象mac.agent，线性层，GRU，线性层，
        # 该对象定义了各个智能体的局部Q网络，即接收观测作为输入，输出智能体各个动作的隐藏层值和Q值。
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    # 运行环境，产生训练样本
    # runner对象中的一个重要属性就是env.multiagentenv.MultiAgentEnv类的环境对象runner.env，即环境，
    # 另一个属性是components.episode_buffer.EpisodeBatch类的episode样本存储器对象runner.batch，
        # 该对象用于以episode为单位存储环境运行所产生的样本。
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner，智能体学习器
    # 该对象的主要作用是依据特定算法对智能体参数进行训练更新
    # 在QMix算法与VDN算法中，均有nn.module类的混合网络learner.mixer，
        # 因此learner对象需要学习的参数包括各个智能体的局部Q网络参数mac.parameters()，以及混合网络参数learner.mixer.parameters()，
        # 两者共同组成了learner.params，然后用优化器learner.optimiser进行优化。
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        with th.no_grad():
            # t_start = time.time()
            episode_batch = runner.run(test_mode=False)
            if episode_batch.batch_size > 0:  # After clearing the batch data, the batch may be empty.
                buffer.insert_episode_batch(episode_batch)
            # print("Sample new batch cost {} seconds.".format(time.time() - t_start))
            episode += args.batch_size_run

        if buffer.can_sample(args.batch_size):
            if args.accumulated_episodes and episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            last_test_T = runner.t_env
            with th.no_grad():
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)

        if args.save_model and (
                runner.t_env - model_save_time >= args.save_model_interval or runner.t_env >= args.t_max):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.log_model_dir, args.unique_token,
                                     str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.log_stat("episode_in_buffer", buffer.episodes_in_buffer, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

    # flush
    sys.stdout.flush()
    time.sleep(10)


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
