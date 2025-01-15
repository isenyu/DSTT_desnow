
import torch.nn.functional as F
import torch.nn as nn
import torch
import time


def maps_2_cubes(x, b, d):
    x_b, x_c, x_h, x_w = x.shape
    x = x.contiguous().view(b, d, x_c, x_h, x_w)

    return x.permute(0, 2, 1, 3, 4)


def maps_2_maps(x, b, d):
    x_b, x_c, x_h, x_w = x.shape
    x = x.contiguous().view(b, d * x_c, x_h, x_w)

    return x



class R_CLSTM_5_stereo(nn.Module):
    def __init__(self, block_channel, use_bn=True):
        super(R_CLSTM_5_stereo, self).__init__()
        # num_features = 64 + block_channel[3] // 32
        num_features = block_channel[3]
        self.Refine = R_3(block_channel, use_bn=use_bn)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor1, input_tensor2, b, d):
        input_tensor1 = maps_2_cubes(input_tensor1, b, d)
        b, c, d, h, w = input_tensor1.shape
        input_tensor2 = maps_2_cubes(input_tensor2, b, d)

        h_state_init = torch.zeros(b, 8, h, w).to('cuda')
        c_state_init = torch.zeros(b, 8, h, w).to('cuda')
        #
        # h_state_init = torch.zeros(b, 8, h, w)
        # c_state_init = torch.zeros(b, 8, h, w)

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner1 = []
        output_inner2 = []
        for t in range(seq_len):
            input_cat1 = torch.cat((input_tensor1[:, :, t, :, :], h_state), dim=1)
            c_state = self.F_t(input_cat1) * c_state + self.I_t(input_cat1) * self.C_t(input_cat1)
            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat1)), 1))
            output_inner1.append(p_depth)

            input_cat2 = torch.cat((input_tensor2[:, :, t, :, :], h_state), dim=1)
            c_state = self.F_t(input_cat2) * c_state + self.I_t(input_cat2) * self.C_t(input_cat2)
            h_state, p_depth2 = self.Refine(torch.cat((c_state, self.Q_t(input_cat2)), 1))
            output_inner2.append(p_depth2)
        layer_output1 = torch.stack(output_inner1, dim=2)
        layer_output2 = torch.stack(output_inner2, dim=2)
        return layer_output1, layer_output2


class R_3(nn.Module):
    def __init__(self, block_channel, use_bn=True):
        super(R_3, self).__init__()

        #num_features = 64 + block_channel[3] // 32 + 8

        num_features = block_channel[3]+8
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        # self.conv2 = nn.Conv2d(
        #     num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

        self.conv2 = nn.Conv2d(
            num_features, 3, kernel_size=5, stride=1, padding=2, bias=True)

        self.convh = nn.Conv2d(
            num_features, 8, kernel_size=3, stride=1, padding=1, bias=True)

        self.use_bn = use_bn

    def forward(self, x):

        x0 = self.conv0(x)
        if self.use_bn:
            x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        if self.use_bn:
            x1 = self.bn1(x1)
        x1 = F.relu(x1)

        h = self.convh(x1)
        out = self.conv2(x1)

        return h, out