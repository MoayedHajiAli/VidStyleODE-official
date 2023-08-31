import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from src.modules.stylegan2.model import ResBlock, ConvLayer, Blur, Upsample, make_kernel


def normalize(v):
    if type(v) == list:
        return [normalize(vv) for vv in v]

    return v * torch.rsqrt((torch.sum(v ** 2, dim=1, keepdim=True) + 1e-8))

class ToSpatialCode(torch.nn.Module):
    def __init__(self, inch, outch, scale):
        super().__init__()
        hiddench = inch // 2
        self.conv1 = ConvLayer(inch, hiddench, 1, activate=True, bias=True)
        self.conv2 = ConvLayer(hiddench, outch, 1, activate=False, bias=True)
        self.scale = scale
        self.upsample = Upsample([1, 3, 3, 1], 2)
        self.blur = Blur([1, 3, 3, 1], pad=(2, 1))
        self.register_buffer('kernel', make_kernel([1, 3, 3, 1]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        for i in range(int(np.log2(self.scale))):
            x = self.upsample(x)
        return x


class BaseNetwork(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self,):
        super().__init__()

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def collect_parameters(self, name):
        params = []
        for m in self.modules():
            if type(m).__name__ == name:
                params += list(m.parameters())
        return params

    def fix_and_gather_noise_parameters(self):
        params = []
        device = next(self.parameters()).device
        for m in self.modules():
            if type(m).__name__ == "NoiseInjection":
                assert m.image_size is not None, "One forward call should be made to determine size of noise parameters"
                m.fixed_noise = torch.nn.Parameter(torch.randn(m.image_size[0], 1, m.image_size[2], m.image_size[3], device=device))
                params.append(m.fixed_noise)
        return params

    def remove_noise_parameters(self, name):
        for m in self.modules():
            if type(m).__name__ == "NoiseInjection":
                m.fixed_noise = None

    def forward(self, x):
        return x
    
class ResnetEncoder(BaseNetwork):
    def __init__(self, netE_num_downsampling_sp, spatial_code_ch, netE_nc_steepness=2.0, netE_scale_capacity=1.0):
        super().__init__()
        self.netE_num_downsampling_sp = netE_num_downsampling_sp
        self.spatial_code_ch = spatial_code_ch
        self.netE_nc_steepness = netE_nc_steepness
        self.netE_scale_capacity = netE_scale_capacity
        # If antialiasing is used, create a very lightweight Gaussian kernel.
        blur_kernel = [1]

        self.add_module("FromRGB", ConvLayer(3, self.nc(0), 1))

        self.DownToSpatialCode = nn.Sequential()
        for i in range(netE_num_downsampling_sp):
            self.DownToSpatialCode.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel,
                         reflection_pad=True)
            )

        nchannels = self.nc(netE_num_downsampling_sp)
        self.add_module(
            "ToSpatialCode",
            nn.Sequential(
                ConvLayer(nchannels, nchannels, 1, activate=True, bias=True),
                ConvLayer(nchannels, spatial_code_ch, kernel_size=1,
                          activate=False, bias=True)
            )
        )


    def nc(self, idx):
        nc = self.netE_nc_steepness ** (5 + idx)
        nc = nc * self.netE_scale_capacity
        return round(nc)

    def forward(self, x, extract_features=False):
        x = self.FromRGB(x)
        midpoint = self.DownToSpatialCode(x)
        sp = self.ToSpatialCode(midpoint)
        sp = normalize(sp)

        if extract_features:
            padded_midpoint = F.pad(midpoint, (1, 0, 1, 0), mode='reflect')
            feature = self.DownToGlobalCode[0](padded_midpoint)
            assert feature.size(2) == sp.size(2) // 2 and \
                feature.size(3) == sp.size(3) // 2
            feature = F.interpolate(
                feature, size=(7, 7), mode='bilinear', align_corners=False)

        if extract_features:
            return sp, feature
        else:
            return sp
