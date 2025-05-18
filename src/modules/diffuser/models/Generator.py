import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from einops import rearrange
import pdb
from torch.distributions import Bernoulli
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, output_shape),
        )
    
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    


class VAE(nn.Module):
    def __init__(self, input_shape, output_shape, latent_dim):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # 解码器
        # 解码器从隐变量生成 zp
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),            
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_shape)
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    # VAE 损失函数
    def loss(self, zp, z, mu, logvar, beta=0.001):
        recon_loss = nn.functional.mse_loss(zp, z, reduction='mean')
        # 计算 KL 散度
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * KLD


# GANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGANGAN
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim_z):
        super(Generator, self).__init__()
        # 生成器
        self.generator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, output_dim_z)
        )

    def forward(self, x):
        """生成器部分，用 obs 和 h 生成向量 zp"""
        zp = self.generator(x)
        return zp

class Discriminator(nn.Module):
    def __init__(self, output_dim_z):
        super(Discriminator, self).__init__()
        # 判别器
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim_z, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )


    def forward(self, z):
        """判别器部分，判断 z 是否为真实数据"""
        validity = self.discriminator(z)
        return validity

