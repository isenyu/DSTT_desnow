import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.SwinIR_stereo import StereoTR
from models.LSTM_stereo import R_CLSTM_5_stereo

def cubes_2_maps(cubes):
    b, c, d, h, w = cubes.shape
    cubes = cubes.permute(0, 2, 1, 3, 4)

    return cubes.contiguous().view(b*d, c, h, w), b, d


def maps_2_cubes(maps, b, d):
    bd, c, h, w = maps.shape
    cubes = maps.contiguous().view(b, d, c, h, w)

    return cubes.permute(0, 2, 1, 3, 4)


def rev_maps(maps, b, d):
    """reverse maps temporarily."""
    cubes = maps_2_cubes(maps, b, d).flip(dims=[2])

    return cubes_2_maps(cubes)[0]


class final_net(nn.Module):
    def __init__(self,  height, width, block_channel=[0, 0, 0, 60], embed_dim=60, window_size=8):
        super(final_net, self).__init__()
        self.stereo_Transformer = StereoTR(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='')
        self.stereo_LSTM = R_CLSTM_5_stereo(block_channel)

    def forward(self, x, x2):
        x, b, d = cubes_2_maps(x)
        x2, b, d = cubes_2_maps(x2)
        x_trans, x2_trans = self.stereo_Transformer(x, x2) #maps  features
        dehaze, dehaze2 = self.stereo_LSTM(x_trans, x2_trans, b, d)  # cube
        return dehaze, dehaze2


#
#
if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size  # 264
    width = (720 // upscale // window_size + 1) * window_size  # 184
    model = final_net(height, width, block_channel=[0, 0, 0, 60], window_size=8)
    #print(model)

    x1 = torch.randn((2, 3, 2, height, width))
    x2 = torch.randn((2, 3, 2, height, width))

    print(x1.shape)
    x1, x2 = model(x1, x2)
    print("after_model", x1.shape)
