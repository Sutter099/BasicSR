# coding=utf-8
# Copyright 2025 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
# (Ported to standalone PyTorch)
""" Standalone PyTorch DINOv3 ConvNeXt model compatible with Hugging Face weights """

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F # Needed for GELU if not using ACT2FN

from basicsr.archs.NAFNet_arch import NAFBlock
from basicsr.utils.registry import ARCH_REGISTRY

logger = logging.getLogger(__name__)

# --- Helper Functions and Classes (DropPath, PlainDropPath, PlainLayerNorm, ConvNeXtLayer, ConvNeXtStage) ---
# --- [Keep the definitions of these classes from the previous response] ---
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()
    output = input.div(keep_prob) * random_tensor
    return output

class PlainDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"

class PlainLayerNorm(nn.LayerNorm):
    """LayerNorm that supports channels_first"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        # Ensure normalized_shape is correctly passed to parent LayerNorm
        if data_format == "channels_first":
             # LayerNorm expects the features dimension as normalized_shape
             # In channels_first, this is the C dimension
             pass # normalized_shape should already be the channel dim
        super().__init__(normalized_shape, eps=eps, elementwise_affine=True) # elementwise_affine=True is default and needed for learnable params
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {data_format}")
        self.data_format = data_format

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_first":
            # NCHW -> NHWC
            features = features.permute(0, 2, 3, 1)
            features = super().forward(features)
            # NHWC -> NCHW
            features = features.permute(0, 3, 1, 2)
        else: # channels_last NHWC
            features = super().forward(features)
        return features


class ConvNeXtLayer(nn.Module):
    """ ConvNeXt Layer (Equivalent to HF's DINOv3ConvNextLayer / Original Block) """
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        layer_norm_eps: float = 1e-6,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # Use PlainLayerNorm with data_format='channels_last' for the main norm
        self.layer_norm = PlainLayerNorm(dim, eps=layer_norm_eps, data_format="channels_last")
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)
        if hidden_act == "gelu":
            self.activation_fn = nn.GELU()
        else:
             self.activation_fn = nn.GELU() # Fallback or add more options
             logger.warning(f"Activation function '{hidden_act}' not explicitly handled, defaulting to GELU.")

        self.pointwise_conv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = PlainDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.depthwise_conv(features)
        features = features.permute(0, 2, 3, 1) # NCHW -> NHWC
        features = self.layer_norm(features)
        features = self.pointwise_conv1(features)
        features = self.activation_fn(features)
        features = self.pointwise_conv2(features)
        features = features * self.gamma
        features = features.permute(0, 3, 1, 2) # NHWC -> NCHW
        features = residual + self.drop_path(features)
        return features

class ConvNeXtStage(nn.Module):
    """ ConvNeXt Stage (Equivalent to HF's DINOv3ConvNextStage) """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stage_idx: int,
        drop_path_rates: List[float],
        layer_scale_init_value: float = 1e-6,
        layer_norm_eps: float = 1e-6,
        hidden_act: str = "gelu",
        num_input_channels: int = 3
    ):
        super().__init__()

        if stage_idx == 0:
            self.downsample_layers = nn.Sequential( # Changed to Sequential for simplicity
                nn.Conv2d(num_input_channels, out_channels, kernel_size=4, stride=4),
                PlainLayerNorm(out_channels, eps=layer_norm_eps, data_format="channels_first"),
            )
        else:
            self.downsample_layers = nn.Sequential( # Changed to Sequential
                PlainLayerNorm(in_channels, eps=layer_norm_eps, data_format="channels_first"),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            )

        self.layers = nn.Sequential( # Changed to Sequential
            *[
                ConvNeXtLayer(
                    dim=out_channels,
                    drop_path=drop_path_rates[i],
                    layer_scale_init_value=layer_scale_init_value,
                    layer_norm_eps=layer_norm_eps,
                    hidden_act=hidden_act,
                )
                for i in range(depth)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.downsample_layers(features)
        features = self.layers(features)
        return features

# --- Standalone Backbone Class (Outputs feature maps) ---
class ConvNeXtBackbone_HFCompat(nn.Module):
    """
    Standalone PyTorch ConvNeXt Backbone compatible with Hugging Face DINOv3 weights.
    Outputs feature maps from specified stages.
    """
    def __init__(
        self,
        num_channels: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        hidden_sizes: List[int] = [96, 192, 384, 768],
        out_indices: Optional[List[int]] = None,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        layer_norm_eps: float = 1e-6,
        hidden_act: str = "gelu",
    ):
        super().__init__()

        if out_indices is None:
            out_indices = [len(depths) - 1]
        self.out_indices = out_indices
        num_stages = len(depths)

        total_depth = sum(depths)
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        current_drop_path_idx = 0
        self.stages = nn.ModuleList() # Use ModuleList to store stages
        in_channels = num_channels
        for i in range(num_stages):
            stage_depth = depths[i]
            stage_drop_path_rates = drop_path_rates[current_drop_path_idx : current_drop_path_idx + stage_depth]
            out_channels = hidden_sizes[i]

            self.stages.append(
                ConvNeXtStage(
                    in_channels=in_channels if i > 0 else num_channels,
                    out_channels=out_channels,
                    depth=stage_depth,
                    stage_idx=i,
                    drop_path_rates=stage_drop_path_rates,
                    layer_scale_init_value=layer_scale_init_value,
                    layer_norm_eps=layer_norm_eps,
                    hidden_act=hidden_act,
                    num_input_channels=num_channels
                )
            )
            in_channels = out_channels
            current_drop_path_idx += stage_depth

        self.hidden_sizes = hidden_sizes
        self.num_features = hidden_sizes

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
             nn.init.trunc_normal_(module.weight, std=.02)
             if module.bias is not None:
                 nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, PlainLayerNorm)):
             nn.init.constant_(module.bias, 0)
             nn.init.constant_(module.weight, 1.0)

    def forward(self, pixel_values: torch.Tensor) -> Union[Tuple[torch.Tensor], List[torch.Tensor]]:
        hidden_states = pixel_values
        output_features = []
        # Store all stage outputs temporarily
        all_stage_outputs = []

        for stage in self.stages:
            hidden_states = stage(hidden_states)
            all_stage_outputs.append(hidden_states)

        # Select features based on out_indices
        for i in self.out_indices:
             if i < len(all_stage_outputs):
                  output_features.append(all_stage_outputs[i])
             else:
                  logger.warning(f"Requested out_indice {i} is out of bounds for {len(all_stage_outputs)} stages.")

        return tuple(output_features)


# --- NEW: Standalone Model Class (Outputs ViT-like pooled + patch tokens) ---
@ARCH_REGISTRY.register()
class ConvNeXtModel_HFCompat(nn.Module):
    """
    Standalone PyTorch ConvNeXt Model compatible with Hugging Face DINOv3 weights.
    Outputs pooled representation and patch tokens (similar to ViT).
    """
    def __init__(
        self,
        num_channels: int = 3,
        depths: List[int] = [3, 3, 9, 3],       # e.g., Tiny
        hidden_sizes: List[int] = [96, 192, 384, 768], # e.g., Tiny
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        layer_norm_eps: float = 1e-6,
        hidden_act: str = "gelu",
        freeze_encoder: bool = True,
        output_hidden_states: bool = False, # Added for compatibility, affects return value slightly
    ):
        super().__init__()
        self.output_hidden_states = output_hidden_states # Store this flag

        num_stages = len(depths)

        total_depth = sum(depths)
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        current_drop_path_idx = 0
        self.stages = nn.ModuleList() # Use ModuleList to store stages
        in_channels = num_channels
        for i in range(num_stages):
            stage_depth = depths[i]
            stage_drop_path_rates = drop_path_rates[current_drop_path_idx : current_drop_path_idx + stage_depth]
            out_channels = hidden_sizes[i]

            self.stages.append(
                ConvNeXtStage(
                    in_channels=in_channels if i > 0 else num_channels,
                    out_channels=out_channels,
                    depth=stage_depth,
                    stage_idx=i,
                    drop_path_rates=stage_drop_path_rates,
                    layer_scale_init_value=layer_scale_init_value,
                    layer_norm_eps=layer_norm_eps,
                    hidden_act=hidden_act,
                    num_input_channels=num_channels
                )
            )
            in_channels = out_channels
            current_drop_path_idx += stage_depth

        self.decoder_blocks = nn.ModuleList()
        last_stage_idx = num_stages - 1
        for i in range(last_stage_idx, 0, -1):
            # i 是当前阶段的索引 (e.g., 3, 2, 1)
            # i-1 是跳跃连接的索引和上一个解码阶段的输出目标通道 (e.g., 2, 1, 0)

            in_dim = hidden_sizes[i]      # 当前解码阶段的输入通道 (来自前一阶段上采样/最深层)
            skip_dim = hidden_sizes[i-1]  # 跳跃连接通道 (也是上采样后期望的通道)

            # 1. PixelShuffle 上采样层 (Upconv)
            # Cin (in_dim) -> Cout * 4 (skip_dim * 4) -> PixelShuffle(2) -> skip_dim
            upconv = nn.Sequential(
                nn.Conv2d(in_dim, skip_dim * 4, 1, bias=False),
                nn.PixelShuffle(2)
            )

            # 2. 融合降维层 (Conv)
            # Concatenation: skip_dim (Upconv out) + skip_dim (Skip connection) = 2 * skip_dim
            conv_dec = nn.Conv2d(skip_dim * 2, skip_dim, 1)

            # 3. NAF 解码块 (Dec Block)
            dec_block = NAFBlock(skip_dim)

            self.decoder_blocks.append(
                nn.ModuleDict({
                    'upconv': upconv,
                    'conv_dec': conv_dec,
                    'dec_block': dec_block
                })
            )

        # --- 最终上采样到原始尺寸 (Stage 0 输出到原图) ---
        upsample_factor = 4 # 假设 Stage 0 输出是 1/4 尺寸
        out_channels = hidden_sizes[0]
        final_intermediate_channels = num_channels * (upsample_factor ** 2)
        self.upconv_final = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_sizes[0], # Stage 0 的输出通道
                out_channels=final_intermediate_channels,
                kernel_size=1,
                bias=False
            ),
            nn.PixelShuffle(upscale_factor=upsample_factor)
        )

        self.hidden_sizes = hidden_sizes # Store just in case
        if freeze_encoder:
            print("Frozen: ConvNeXt encoder...")
            for param in self.stages.parameters():
                param.requires_grad = False

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights (matching HF style) """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
             nn.init.trunc_normal_(module.weight, std=.02)
             if module.bias is not None:
                 nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, PlainLayerNorm)):
             nn.init.constant_(module.bias, 0)
             nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
    ) -> Dict[str, Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]]:
        """
        Mimics HF DINOv3ConvNextModel forward pass.

        Returns:
            A dictionary containing:
            - 'last_hidden_state': Tensor of shape (B, 1 + H*W, C) after final LayerNorm.
            - 'pooler_output': Tensor of shape (B, C), the pooled representation before final LayerNorm
                               (or after, depending on exact HF implementation nuance - here using after LN).
            - 'hidden_states': Tuple of tensors (input + output of each stage), if requested.
        """
        hidden_states = pixel_values
        hidden_states_list = []

        # --- Pass through stages ---
        for stage in self.stages:
            hidden_states = stage(hidden_states)
            hidden_states_list.append(hidden_states)

        current_features = hidden_states_list[-1]

        skip_connections = hidden_states_list[:-1]
        reversed_skip_connections = skip_connections[::-1]

        # N-1 个解码阶段
        for i, dec_mod in enumerate(self.decoder_blocks):
            skip_feature = reversed_skip_connections[i]

            # 1. PixelShuffle 上采样 (Upconv)
            D_up = dec_mod['upconv'](current_features)

            # 2. 跳跃连接和融合
            # 确保跳跃连接特征图的尺寸与上采样特征图的尺寸匹配 (U-Net)
            # 在实际操作中，您可能需要确保 D_up 和 skip_feature 的 H, W 尺寸严格一致
            # 如果不一致，可能需要进行裁剪或插值

            D_fused = torch.cat([D_up, skip_feature], dim=1)  # 拼接
            D_fused = dec_mod['conv_dec'](D_fused)            # 降维

            # 3. NAFBlock
            current_features = dec_mod['dec_block'](D_fused)  # NAFBlock

        output = self.upconv_final(current_features)

        return output


# --- Configuration Dictionary ---
convnext_hf_configs = {
    "tiny": dict(depths=[3, 3, 9, 3], hidden_sizes=[96, 192, 384, 768]),
    "small": dict(depths=[3, 3, 27, 3], hidden_sizes=[96, 192, 384, 768]),
    "base": dict(depths=[3, 3, 27, 3], hidden_sizes=[128, 256, 512, 1024]),
    "large": dict(depths=[3, 3, 27, 3], hidden_sizes=[192, 384, 768, 1536]),
}

# # --- Example Usage ---
# if __name__ == "__main__":
#     model = ConvNeXtModel_HFCompat(
#         output_hidden_states=True # Also request intermediate states
#     )
#     dummy_input = torch.randn(2, 3, 256, 256)
#     output = model(dummy_input)
#     print("input shape: ", dummy_input.shape)
#     print("output shape: ", output.shape)
