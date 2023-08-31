import torch.nn as nn
import torch
import numpy as np
from src.modules.stylegan2.stylegan_human import dnnlib as fashion_dnnlib
from src.modules.stylegan2.stylegan_human.model import Generator as FashionGenerator
from src.modules.stylegan2.model import Generator


def get_generator_params(pkl_obj):
    if 'g' in pkl_obj:
        return pkl_obj['g']
    elif 'G_ema' in pkl_obj:
        return pkl_obj['G_ema']
    elif 'g_ema' in pkl_obj:
        return pkl_obj['g_ema']
    else:
        raise "cannot find appropriate key"

class StyleGAN2Fashion(nn.Module):
    def __init__(self, pkl_file):
        super().__init__()
        with fashion_dnnlib.util.open_url(pkl_file) as f:
            self.G = fashion_dnnlib.legacy.load_network_pkl(f)['G_ema'].eval()
            self.G = self.G.float()

        for parameter in self.G.parameters():
            parameter.requires_grad = False

    def forward(self, w):
        return self.G.synthesis(w, noise_mode='const', force_fp32=True)

    def synthesis(self, device):
        label = torch.zeros([1, self.G.c_dim]).to(device)
        z = torch.from_numpy(np.random.randn(1, self.G.z_dim)).to(device)
        w = self.G.mapping(z, label)
        return self.forward(w)


class StyleGAN2Fashion_FT(nn.Module):
    # finetuned stylegan human
    def __init__(self, pkl_file):
        super().__init__()
        self.G = FashionGenerator(1024, 512, 8)
        pkl_obj = get_generator_params(torch.load(pkl_file, map_location='cpu'))
        self.G.load_state_dict(pkl_obj, strict=True)
        self.G.float().eval()

        for parameter in self.G.parameters():
            parameter.requires_grad = False

    def forward(self, w):
        return self.G([w], input_is_latent=True, randomize_noise=False)[0]

class StyleGAN2Ravdess(nn.Module):
    def __init__(self, pkl_file):
        super().__init__()
        self.G = Generator(1024, 512, 8)
        pkl_obj = get_generator_params(torch.load(pkl_file, map_location='cpu'))
        self.G.load_state_dict(pkl_obj, strict=True)
        self.G.float().eval()

        for parameter in self.G.parameters():
            parameter.requires_grad = False

    def forward(self, w):
        return self.G([w], input_is_latent=True, randomize_noise=False)[0]
