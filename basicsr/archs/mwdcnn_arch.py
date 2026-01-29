'''
    This model is dynamic conv + wavelet transform + Residual dense block callde DWD
'''

import torch
import torch.nn as nn
from model_common import common
from model_common.WRB import WRB
from model_common.RDB import RDB
from basicsr.utils.registry import ARCH_REGISTRY


class WMDCNN(nn.Module):
    def __init__(self, n_colors=3, n_feats=64, growth_rate=32, rdb_num_layers=4, debug=False, conv=common.default_conv, **kwargs):
        super(WMDCNN, self).__init__()

        kernel_size = 5
        dynamic_conv = common.dynamic_conv

        self.conv1 = conv(n_colors, n_feats, kernel_size)  # conv1

        self.dy_conv_block = nn.Sequential(
            dynamic_conv(n_feats, n_feats, kernel_size),
            nn.ReLU(True)
        )

        self.conv_block1 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(True)
        )

        self.WRB1 = WRB(n_feats, growth_rate, rdb_num_layers, debug=debug)
        self.WRB2 = WRB(n_feats, growth_rate, rdb_num_layers, debug=debug)

        self.RDB_1 = nn.Sequential(
            RDB(n_feats, growth_rate, rdb_num_layers),
            nn.ReLU(True)
        )

        self.RDB_2 = nn.Sequential(
            RDB(n_feats, growth_rate, rdb_num_layers),
            nn.ReLU(True)
        )

        self.conv_block2 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(True)
        )

        self.conv2 = conv(n_feats, n_colors, kernel_size)  # conv1

        self.seq = nn.Sequential(
            self.conv1,
            self.dy_conv_block,
            self.conv_block1,
            self.WRB1,
            self.WRB2
        )

    def forward(self, x):

        y = x
        
        out1 = self.seq(x)

        out2 = self.RDB_1(out1)
        out3 = self.RDB_2(out2)

        out4 = out1 + out2 + out3

        out5 = self.conv_block2(out4)
        out = self.conv2(out5)

        return y - out

# Safe registration
if 'WMDCNN' not in ARCH_REGISTRY:
    ARCH_REGISTRY.register(WMDCNN)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_feats', type=int, default=64)
    parser.add_argument('--growth_rate', type=int, default=32)
    parser.add_argument('--RDB_num_layers', type=int, default=4)
    parser.add_argument('--n_colors', type=int, default=3)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    model = WMDCNN(
        n_colors=args.n_colors,
        n_feats=args.n_feats,
        growth_rate=args.growth_rate,
        RDB_num_layers=args.RDB_num_layers,
        debug=args.debug
    )
    print(model)

    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
