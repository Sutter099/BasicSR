'''
    Wavelets Trasform + Residual dense block + Relu Block
'''


import torch
import torch.nn as nn
from model_common.RDB import RDB
from pytorch_wavelets import DWTForward, DWTInverse   # (or import DWT, IDWT)


class WRB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, wave='db3', debug=False):
        super(WRB, self).__init__()

        self.debug = debug
        n_feats = in_channels*4
        growth_rate = growth_rate
        num_layers = num_layers

        self.RDB = RDB(n_feats, growth_rate, num_layers)

        # self.conv1 = nn.Conv2d(in_channels*4, in_channels*2, kernel_size=1)     #
        self.conv2 = nn.Conv2d(n_feats+num_layers*growth_rate, in_channels*4, kernel_size=1)

        # J为分解的层次数,wave表示使用的变换方法
        self.WTF = DWTForward(J=1, mode='zero', wave=wave)  # Accepts all wave types available to PyWavelets
        self.WTI = DWTInverse(mode='zero', wave=wave)

    def forward(self, x):
        batch_size, _, h, w = x.shape

        if h % 2 == 1 and w % 2 == 1:
            pad = nn.ReplicationPad2d(1)
            x = pad(x)
            x = x[:, :, 0:-1, 0:-1]
        elif h % 2 == 1:
            pad = nn.ReplicationPad2d(1)
            x = pad(x)
            x = x[:, :, 0:-1, 1:-1]
        elif w % 2 == 1:
            pad = nn.ReplicationPad2d(1)
            x = pad(x)
            x = x[:, :, 1:-1, 0:-1]

        yl, yh = self.WTF(x)
        yh = yh[0]  # 返回的yh是一个list
        fh, fw = yh.shape[-2], yh.shape[-1]
        yh = yh.view(batch_size, -1, fh, fw)

        out = torch.cat((yl, yh), 1)

        out2 = self.RDB(out, False)

        if self.debug:
            print("Output2.shape:{}".format(out2.shape)) # 320

        out3 = self.conv2(out2)

        if self.debug:
            print("Output3.shape:{}".format(out3.shape)) #256


        out3 = out3+out

        yl = out3[:, 0:(yl.shape[1]), :, :]
        yh = out3[:, yl.shape[1]:, :, :].view(batch_size, -1, 3, fh, fw)
        yh = [yh, ]
        out = self.WTI((yl, yh))

        if h % 2 == 1 and w % 2 == 1:
            out = out[:, :, 1:, 1:]
        elif h % 2 == 1:
            out = out[:, :, 1:, :]
        elif w % 2 == 1:
            out = out[:, :, :, 1:]

        return out
