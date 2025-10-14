# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import LayerNorm2d, to_2tuple, trunc_normal_
from basicsr.archs.local_arch import Local_Base
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from pytorch_wavelets import DWTForward, DWTInverse

from basicsr.archs.mpma import *

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class WaveletDenoiseBlock(nn.Module):
    def __init__(self, c, num_blks=1, wave='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)
        self.iwt = DWTInverse(mode='zero', wave=wave)

        # Create separate denoisers for Horizontal and Vertical sub-bands
        # These denoisers are applied to the LH and HL components
        self.denoiser_h = nn.Sequential(*[NAFBlock(c) for _ in range(num_blks)])
        self.denoiser_v = nn.Sequential(*[NAFBlock(c) for _ in range(num_blks)])

    def forward(self, x):
        # Decompose the input feature map into wavelet sub-bands
        Yl, Yh = self.dwt(x) # Yl shape: (B, C, H/2, W/2)
                             # Yh[0] shape: (B, C, 3, H/2, W/2) -> LH, HL, HH

        # Extract the three detail components (LH, HL, HH)
        lh, hl, hh = torch.unbind(Yh[0], dim=2)

        # Apply specialized denoisers to the horizontal and vertical bands
        denoised_lh = self.denoiser_h(lh)
        denoised_hl = self.denoiser_v(hl)

        # Reconstruct the detail components list
        # We can choose to leave hh unprocessed as stripe noise is less dominant here
        reconstructed_Yh = [torch.stack([denoised_lh, denoised_hl, hh], dim=2)]

        # Reconstruct the feature map from the processed sub-bands
        output_feature = self.iwt((Yl, reconstructed_Yh))

        return output_feature

@ARCH_REGISTRY.register()
class NAFMamba(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=[], enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()

        # Decoder for Detail Reconstruction (the main one)
        self.decoders_detail = nn.ModuleList()
        self.ups_detail = nn.ModuleList()

        # Decoder for Low-Frequency Noise
        self.decoders_lf = nn.ModuleList()
        self.ups_lf = nn.ModuleList()
        self.ending_lf = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        # Decoder for Stripe Noise (Wavelet-based)
        self.decoders_stripe = nn.ModuleList()
        self.ups_stripe = nn.ModuleList()
        self.ending_stripe = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        # self.middle_blks = nn.ModuleList() # Removed, will be replaced by MambaIRv2's forward_features
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = None
        self.middle_in_channels = chan
        self.middle_blk_num = middle_blk_num

        # Build the three parallel decoders
        for num in dec_blk_nums:
            # Upsampling is shared for simplicity, or can be separate
            up_module = nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            )
            self.ups_detail.append(up_module)
            self.ups_lf.append(up_module)
            self.ups_stripe.append(up_module)

            chan = chan // 2

            # Detail Decoder Branch
            self.decoders_detail.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

            # Low-Frequency Decoder Branch (can be simpler, e.g., fewer blocks)
            self.decoders_lf.append(nn.Sequential(*[NAFBlock(chan, FFN_Expand=1) for _ in range(num // 2 + 1)]))
            # self.decoders_lf.append(nn.Sequential(*[NAFBlock(chan, FFN_Expand=1) for _ in range(num)]))

            # Stripe Decoder Branch (using Wavelet Denoising Block)
            self.decoders_stripe.append(WaveletDenoiseBlock(chan))

        self.padder_size = 2 ** len(self.encoders)

    # Helper function to run a decoder path
    def run_decoder(self, decoders, upsamplers, x, skips):
        for decoder, up, skip in zip(decoders, upsamplers, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)
        return x

    def forward(self, inp):
        B, C, H, W = inp.shape

        # print(f"before padding: {inp.shape}") # INFO:

        inp = self.check_image_size(inp)

        # print(f"input size: {inp.shape}") # INFO:

        x = self.intro(inp)

        encs = []

        # print(f"after intro: {x.shape}") # INFO:

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            # print(f"After downsampling: {x.shape}") # INFO:

        # --- Lazy Initialization of the Bottleneck ---
        if self.middle_blks is None:
            b, c, h, w = x.shape
            self.middle_blks = nn.ModuleList()
            for num in self.middle_blk_num:
                self.middle_blks.append(
                    nn.Sequential(*[MPMABlock(in_channels=self.middle_in_channels, image_size=h) for _ in range(num)])
                )
            self.middle_blks = self.middle_blks.to(x.device) # Move to the correct device

        for blk in self.middle_blks:
            x = blk(x)

        bottle_out = x

        # print(f"After bottleneck: {x.shape}") # INFO:

        # Detail reconstruction path
        x_detail = self.run_decoder(self.decoders_detail, self.ups_detail, bottle_out, encs)
        pred_detail = self.ending(x_detail)

        final_clean_image = inp + pred_detail

        # Or if you use a global skip connection from input
        # final_clean_image = inp - pred_lf_noise - pred_stripe_noise

        # For training, you should return all intermediate predictions for the loss function
        # return final_clean_image, pred_lf_noise, pred_stripe_noise

        # For inference, just return the final image
        # Note: The global skip connection 'x + inp' might need to be re-evaluated
        # in this multi-head setup. Let's return the combined prediction directly.

        return final_clean_image[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, NAFMamba):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
