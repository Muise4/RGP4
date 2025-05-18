import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto

class Config(ParamsProto):
    def __init__(self, args):
        super(Config, self).__init__()
        self.args = args
        # misc
    seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bucket = '/home/xizhen/data/weights/'
    # dataset = 'hopper-medium-expert-v2'
    bucket = None
    dataset = None


    # 是否预测噪声
    # 对于di_hpn_qmix来说,因为是直接预测的过了PI网络之后的状态,所以都是MSE损失,
    # 而预测MSE损失又很明显预测噪声更合适,所以不用预测x0,也因此没必要改后面的损失代码
    predict_epsilon = False
    ## model
    # model = 'models.TemporalUnet' #现在Unet是Unet
    model = 'models.Transformer' #Transfomer改成了MLP
    diffusion = 'models.GaussianInvDynDiffusion'
    # BCE很费时，可能代码可以优化改下？但transformer更加费时得多得多
    if model == 'models.TemporalUnet':
        n_diffusion_steps = 10
    else :
        n_diffusion_steps = 20
    # if predict_epsilon == False :
    #         # loss_type = 'l2'
    # else :
        
    horizon = 1
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    dim_mults = (1, 4, 8)
    returns_condition = False
    calc_energy=False
    dim=128
    condition_dropout=0.25
    condition_guidance_w = 1.2
    test_ret=0.9
    renderer = 'utils.MuJoCoRenderer'

    ## dataset
    loader = 'datasets.SequenceDataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = False
    discount = 0.99
    max_path_length = 1000
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False
    termination_penalty = -100
    returns_scale = 400.0 # Determined using rewards from the dataset

    ## training
    n_steps_per_epoch = 10000
    n_train_steps = 1e6
    batch_size = 32
    learning_rate = 2e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 10000
    sample_freq = 10000
    n_saves = 5
    loss_type = 'l2'
    save_parallel = False
    n_reference = 8
    save_checkpoints = False
