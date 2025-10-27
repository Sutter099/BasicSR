# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.NAFNet_arch import NAFBlock

logger = logging.getLogger("dinov3_denoiser")


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    r"""
    Code adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.pyConvNeXt

    A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        patch_size (int | None): Pseudo patch size. Used to resize feature maps to those of a ViT with a given patch size. If None, no resizing is performed
    """

    def __init__(
        self,
        # original ConvNeXt arguments
        in_chans: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        # DINO arguments
        patch_size: Optional[int] = None,
        **ignored_kwargs,
    ):
        super().__init__()
        # TODO: remove?
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        # ==== ConvNeXt's original init =====
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.dims = dims
        self.patch_size = patch_size # Store patch_size if needed later
        # ==== End of ConvNeXt's original init =====

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        if isinstance(module, LayerNorm):
            module.weight = nn.Parameter(torch.ones(module.normalized_shape))
            module.bias = nn.Parameter(torch.zeros(module.normalized_shape))
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    # def forward_features(self, x: Union[Tensor, List[Tensor]], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
    def forward_features_original_dino_output(self, x: Union[Tensor, List[Tensor]], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        output = []
        for x, masks in zip(x_list, masks_list):
            h, w = x.shape[-2:]
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            x_pool = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
            x = torch.flatten(x, 2).transpose(1, 2)

            # concat [CLS] and patch tokens as (N, HW + 1, C), then normalize
            x_norm = self.norm(torch.cat([x_pool.unsqueeze(1), x], dim=1))
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_storage_tokens": x_norm[:, 1 : self.n_storage_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )

        return output

    def forward(self, x):
        # Forward through stages to get features at different resolutions
        features = []
        current_x = x
        for i in range(4):
            current_x = self.downsample_layers[i](current_x)
            current_x = self.stages[i](current_x)
            features.append(current_x)
        # Return features from all stages for potential U-Net style skip connections
        return features

convnext_sizes = {
    "tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    "small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    "base": dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    "large": dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}

class UNetDecoder(nn.Module):
    def __init__(self, encoder_dims: List[int], output_chans: int = 3):
        super().__init__()
        # encoder_dims = [96, 192, 384, 768] for tiny

        # --- Upsampling Blocks ---
        # Start from the deepest feature
        self.upconv3 = nn.Sequential(
            nn.Conv2d(encoder_dims[3], encoder_dims[3] * 2, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.dec_block3 = NAFBlock(encoder_dims[3] // 2)

        self.upconv2 = nn.Sequential(
            nn.Conv2d(encoder_dims[2], encoder_dims[2] * 2, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.dec_block2 = NAFBlock(encoder_dims[2] // 2)

        self.upconv1 = nn.Sequential(
            nn.Conv2d(encoder_dims[1], encoder_dims[1] * 2, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.dec_block1 = NAFBlock(encoder_dims[1] // 2)

        # --- Final Upsampling to original size ---
        upsample_factor = 4
        intermediate_channels = output_chans * (upsample_factor ** 2)
        self.upconv_final = nn.Sequential(
            nn.Conv2d(
                in_channels=encoder_dims[0],
                out_channels=intermediate_channels,
                kernel_size=1,
                bias=False
            ),
            nn.PixelShuffle(upscale_factor=upsample_factor)
        )

    def forward(self, features: List[Tensor]):
        f1, f2, f3, f4 = features # features[0] to features[3]

        d3 = self.upconv3(f4) # Upsample deep feature
        d3_out = self.dec_block3(d3 + f3)

        d2 = self.upconv2(d3_out)
        d2_out = self.dec_block2(d2 + f2)

        d1 = self.upconv1(d2_out)
        d1_out = self.dec_block1(d1 + f1)

        output = self.upconv_final(d1_out)

        return output

# --- Combined Denoising Model ---
@ARCH_REGISTRY.register()
class DINOv3ConvNeXtDenoiser(nn.Module):
    """
    Combines DINOv3 ConvNeXt Backbone with a Decoder for Image Denoising.
    Expects (B, C, H, W) input and outputs (B, C, H, W).
    """
    def __init__(
        self,
        arch_name: str = 'tiny',
        in_chans: int = 3,
        output_chans: int = 3,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        freeze_backbone: bool = True,
        **kwargs
    ):
        super().__init__()

        try:
            size_dict = convnext_sizes[arch_name]
        except KeyError:
            raise NotImplementedError(f"ConvNeXt size '{arch_name}' not recognized.")

        self.backbone = ConvNeXt(
            in_chans=in_chans, **size_dict, drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value, **kwargs
        )
        encoder_dims = size_dict['dims']
        if freeze_backbone:
            logger.info("Freezing DINOv3 ConvNeXt backbone weights.")
            print("freezing")
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.decoder = UNetDecoder(encoder_dims=encoder_dims, output_chans=output_chans)

        logger.info(f"Initialized DINOv3ConvNeXtDenoiser with {arch_name} backbone.")

    def forward(self, x):
        input_h, input_w = x.shape[-2:]
        features = self.backbone(x)
        reconstructed_image = x + self.decoder(features)

        output_h, output_w = reconstructed_image.shape[-2:]
        if output_h != input_h or output_w != input_w:
             reconstructed_image = F.interpolate(
                 reconstructed_image, size=(input_h, input_w),
                 mode='bilinear', align_corners=False
             )
        return reconstructed_image

def get_convnext_arch(arch_name):
    size_dict = None
    query_sizename = arch_name.split("_")[1]
    try:
        size_dict = convnext_sizes[query_sizename]
    except KeyError:
        raise NotImplementedError("didn't recognize vit size string")

    return partial(
        ConvNeXt,
        **size_dict,
    )

# if __name__ == "__main__":
#     # Create a DINOv3ConvNeXtDenoiser model (tiny version)
#     denoiser = DINOv3ConvNeXtDenoiser(arch_name='tiny', in_chans=3, output_chans=3)
#     print(denoiser)
#
#     # Create a dummy input tensor (batch_size, channels, height, width)
#     dummy_input = torch.randn(2, 3, 256, 256)
#
#     # Forward pass
#     output = denoiser(dummy_input)
#     print("Input shape:", dummy_input.shape)
#     print("Output shape:", output.shape) # Should match input shape
#
#     # Check number of parameters
#     num_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
#     print(f"Total trainable parameters: {num_params / 1e6:.2f} M")
