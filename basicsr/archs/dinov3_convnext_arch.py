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

        # Final LayerNorm (applied after pooling and concatenation)
        # Note: HF applies it in channel_last format implicitly after transpose
        self.layer_norm = nn.LayerNorm(hidden_sizes[-1], eps=layer_norm_eps)
        # Pooling layer
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.hidden_sizes = hidden_sizes # Store just in case

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
        output_hidden_states: Optional[bool] = None # Allow overriding init flag
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
        actual_output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states

        hidden_states = pixel_values
        all_hidden_states_list = [hidden_states] if actual_output_hidden_states else None

        # --- Pass through stages ---
        for stage in self.stages:
            hidden_states = stage(hidden_states)
            if actual_output_hidden_states:
                all_hidden_states_list.append(hidden_states)

        # --- Mimic HF Output Formatting ---
        # 1. Global Average Pooling for "CLS" token
        # Input to pool is last stage output: (B, C_last, H_last, W_last)
        pooled_output_nhwc = self.pool(hidden_states) # (B, C_last, 1, 1)

        # 2. Flatten pooled and patch features
        # (B, C, 1, 1) -> (B, 1, C)
        pooled_output_tokens = pooled_output_nhwc.flatten(2).transpose(1, 2)
        # (B, C, H, W) -> (B, H*W, C)
        patch_tokens = hidden_states.flatten(2).transpose(1, 2)

        # 3. Concatenate pooled ("CLS") and patch tokens
        # Result shape: (B, 1 + H*W, C)
        concatenated_tokens = torch.cat([pooled_output_tokens, patch_tokens], dim=1)

        # 4. Apply final LayerNorm
        last_hidden_state = self.layer_norm(concatenated_tokens)

        # Pooler output is typically the first token (pooled representation)
        pooler_output = last_hidden_state[:, 0]

        return_dict = {
             "last_hidden_state": last_hidden_state,
             "pooler_output": pooler_output,
             "hidden_states": tuple(all_hidden_states_list) if actual_output_hidden_states else None
        }
        return return_dict


# --- Configuration Dictionary ---
convnext_hf_configs = {
    "tiny": dict(depths=[3, 3, 9, 3], hidden_sizes=[96, 192, 384, 768]),
    "small": dict(depths=[3, 3, 27, 3], hidden_sizes=[96, 192, 384, 768]),
    "base": dict(depths=[3, 3, 27, 3], hidden_sizes=[128, 256, 512, 1024]),
    "large": dict(depths=[3, 3, 27, 3], hidden_sizes=[192, 384, 768, 1536]),
}

# --- Factory functions ---
def create_dinov3_convnext_backbone(
    model_name: str = "tiny",
    pretrained: bool = False,
    weights_path: Optional[str] = None,
    strict_load: bool = True,
    out_indices: Optional[List[int]] = None,
    **kwargs
) -> ConvNeXtBackbone_HFCompat:
    # --- [Keep the factory function for Backbone from previous response] ---
    if model_name not in convnext_hf_configs:
        raise ValueError(f"Unknown model_name: {model_name}. Available: {list(convnext_hf_configs.keys())}")

    config = convnext_hf_configs[model_name].copy() # Use copy
    config.update(kwargs) # Allow overriding config defaults via kwargs

    if out_indices is None:
         # Default for backbone: output features from all stages for U-Net
         out_indices = list(range(len(config['depths'])))
         # Or maybe just the last one? Depends on usage. Let's default to all.
         # out_indices = [len(config['depths']) - 1]

    model = ConvNeXtBackbone_HFCompat(out_indices=out_indices, **config)

    if pretrained and weights_path:
        logger.info(f"Loading pretrained weights for Backbone from: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        # Check if state_dict is nested (e.g. from HF save)
        if 'model' in state_dict: # Common key used by HF .save_pretrained
             state_dict = state_dict['model']
        elif 'state_dict' in state_dict: # Another common pattern
             state_dict = state_dict['state_dict']
        # Add more checks if needed based on how the .pth was saved

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): # Handle DDP
                cleaned_state_dict[k[7:]] = v
            # Specific HF to standalone key renaming (if any needed, seems names match well)
            # Example: k = k.replace('hf_layer_name', 'standalone_layer_name')
            else:
                cleaned_state_dict[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=strict_load)
        if missing_keys: logger.warning(f"Backbone Missing keys: {missing_keys}")
        if unexpected_keys: logger.warning(f"Backbone Unexpected keys: {unexpected_keys}")
    elif pretrained and not weights_path:
         logger.warning("Backbone `pretrained=True` but no `weights_path`. Randomly initialized.")

    return model


def create_dinov3_convnext_model(
    model_name: str = "tiny", # 'tiny', 'small', 'base', 'large'
    pretrained: bool = False,
    weights_path: Optional[str] = None,
    strict_load: bool = True,
    **kwargs # Pass other args like drop_path_rate, layer_scale_init_value, output_hidden_states
) -> ConvNeXtModel_HFCompat:
    """
    Creates a standalone DINOv3 ConvNeXt model (ViT-like output) compatible with HF weights.
    """
    if model_name not in convnext_hf_configs:
        raise ValueError(f"Unknown model_name: {model_name}. Available: {list(convnext_hf_configs.keys())}")

    config = convnext_hf_configs[model_name].copy()
    config.update(kwargs)

    model = ConvNeXtModel_HFCompat(**config)

    if pretrained and weights_path:
        logger.info(f"Loading pretrained weights for Model from: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        # Handle potential nesting
        if 'model' in state_dict: state_dict = state_dict['model']
        elif 'state_dict' in state_dict: state_dict = state_dict['state_dict']

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                cleaned_state_dict[k[7:]] = v
            else:
                cleaned_state_dict[k] = v
                
        # --- Specific Key Adjustment for ConvNeXtModel vs Backbone ---
        # The HF DINOv3ConvNextModel includes a final 'layer_norm'. The weights file likely
        # contains keys like 'layer_norm.weight' and 'layer_norm.bias' at the top level.
        # Our ConvNeXtModel_HFCompat also has self.layer_norm.
        # The ConvNeXtBackbone_HFCompat *does not* have this final layer_norm.
        # The loading logic here *should* match the keys directly for ConvNeXtModel_HFCompat.

        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=strict_load)
        if missing_keys: logger.warning(f"Model Missing keys: {missing_keys}")
        # Unexpected keys might include things like 'pool' if it's not saved explicitly,
        # or if the loaded state dict is actually from the Backbone variant.
        if unexpected_keys: logger.warning(f"Model Unexpected keys: {unexpected_keys}")
    elif pretrained and not weights_path:
         logger.warning("Model `pretrained=True` but no `weights_path`. Randomly initialized.")

    return model

# # 加载模型权重
# # 创建 tiny 模型，输出最后两个 stage 的特征
# backbone = create_dinov3_convnext_model(
#     model_name="tiny",
#     # out_indices=[2, 3],
#     pretrained=True,
#     weights_path="/home/blu/work/repos/BasicSR/working/dinov3_pth/dinov3-convnext-tiny-pretrain-lvd1689m-model.pth", # 使用上面保存的 .pth
#     strict_load=True # 使用 True 检查是否所有 backbone 键都匹配
# )

# 现在 backbone 可以作为 nn.Module 在你的 BasicSR 架构中使用了
# 例如: self.backbone = backbone

# # --- Example Usage ---
# if __name__ == "__main__":
#     # --- Test Backbone ---
#     print("--- Testing Backbone ---")
#     backbone = create_dinov3_convnext_backbone(
#         model_name="tiny",
#         out_indices=[0, 1, 2, 3], # Get all stage outputs
#         # pretrained=True, weights_path="path/to/pytorch_model.pth"
#     )
#     dummy_input = torch.randn(2, 3, 224, 224)
#     features = backbone(dummy_input)
#     print(f"Number of output feature maps: {len(features)}")
#     for i, feat in enumerate(features):
#         print(f"Feature map {i} shape: {feat.shape}") # Should be H/4, H/8, H/16, H/32
#
#     # --- Test Model ---
#     print("\n--- Testing Model ---")
#     model = create_dinov3_convnext_model(
#          model_name="tiny",
#          # pretrained=True, weights_path="path/to/pytorch_model.pth",
#          output_hidden_states=True # Also request intermediate states
#     )
#     # print("\nKeys expected by model:")
#     # print(list(model.state_dict().keys()))
#     dummy_input = torch.randn(2, 3, 256, 256)
#     output = model(dummy_input)
#
#     print("Model Output Keys:", output.keys())
#     print("Last Hidden State shape:", output['last_hidden_state'].shape) # Should be (B, 1 + HW/32^2, C)
#     print("Pooler Output shape:", output['pooler_output'].shape) # Should be (B, C)
#     if output['hidden_states']:
#          print("Number of hidden_states:", len(output['hidden_states'])) # Should be num_stages + 1
#          print("Shape of first hidden_state (input):", output['hidden_states'][0].shape)
#          print("Shape of last hidden_state (before pool/norm):", output['hidden_states'][-1].shape)
