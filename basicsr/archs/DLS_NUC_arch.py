import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

'''
@article{He-DLS-NUC-2018,
  author = {Zewei He and Yanpeng Cao and Yafei Dong and Jiangxin Yang and Yanlong Cao and Christel-L"{o}ic Tisse},
  journal = {Appl. Opt.},
  number = {18},
  pages = {D155--D164},
  title = {Single-image-based nonuniformity correction of uncooled long-wave infrared detectors: a deep-learning approach},
  volume = {57},
  month = {Jun},
  year = {2018},
  doi = {10.1364/AO.57.00D155},
}
'''

@ARCH_REGISTRY.register()
class DLS_NUC(nn.Module):
    """
    PyTorch implementation of the DLS-NUC model (Case 4 architecture).
    Based on 'Single-image-based nonuniformity correction of uncooled long-wave infrared detectors: a deep-learning approach'.
    """

    def __init__(self, num_channels=1, num_features=64, scale=3, kernel_size=3):
        super(DLS_NUC, self).__init__()

        # Calculate padding to keep spatial dimensions same (for 3x3 -> pad 1)
        padding = (kernel_size - 1) // 2

        # --- Branch 1: Feature Extraction ---
        # Note: MATLAB Case 4 code does not show a ReLU after the first convolution
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size, padding=padding)

        # --- Branch 2: Multi-scale Feature Extraction ---
        # Downsampling
        # MatConvNet vl_nnpool defaults to MaxPool
        self.pool = nn.MaxPool2d(kernel_size=scale, stride=scale)

        # Deep nonlinear mapping (7 layers of Conv+ReLU)
        # Corresponds to weights{2} through weights{8}
        body_layers = []
        for _ in range(7):
            body_layers.append(
                nn.Conv2d(num_features, num_features, kernel_size, padding=padding)
            )
            body_layers.append(nn.ReLU(inplace=True))

        # Layer 9 (weight{9}) - Conv without ReLU immediately?
        # MATLAB code: convfea9 = vl_nnconv(convfea,weight{9},...) -> then Upsample
        body_layers.append(
            nn.Conv2d(num_features, num_features, kernel_size, padding=padding)
        )

        self.body = nn.Sequential(*body_layers)

        # Upsampling (weight{10})
        # vl_nnconvt with 'Upsample', scale.
        # We use ConvTranspose2d. To match dimensions exactly (Output = Input * scale),
        # we assume kernel_size=scale or configured appropriately.
        # With kernel_size=3, stride=3, padding=0 -> Out = (In-1)*3 + 3 = 3*In (Matches)
        self.upsample = nn.ConvTranspose2d(
            num_features, num_features, kernel_size, stride=scale, padding=0
        )

        # --- Fusion & Reconstruction ---
        # Layer 11 (weight{11})
        # Concatenates output of conv1 (branch 1) and upsampled branch 2
        self.final_conv = nn.Conv2d(
            num_features * 2, num_channels, kernel_size, padding=padding
        )

    def forward(self, x):
        # x: [Batch, Channels, Height, Width]

        # Branch 1
        x1 = self.conv1(x)
        # convfea1 = convfea (No ReLU in Case 4 source)

        # Branch 2
        x2 = self.pool(x1)
        x2 = self.body(x2)
        x2_up = self.upsample(x2)

        # Handle potential size mismatches due to padding/pooling
        if x2_up.size() != x1.size():
            x2_up = F.interpolate(
                x2_up, size=x1.shape[2:], mode="bilinear", align_corners=False
            )

        # Concatenate along channel dimension
        # MATLAB: cat(3, convfea1, convfea10) -> Dim 3 is channels in HWC
        # PyTorch: Dim 1 is channels in BCHW
        cat_feat = torch.cat([x1, x2_up], dim=1)

        # Estimate the nonuniformity (noise)
        noise_map = self.final_conv(cat_feat)

        # Residual correction: Output = Input - Noise
        return x - noise_map


if __name__ == "__main__":
    # Simple test to verify shape consistency
    model = DLS_NUC(scale=3)
    # Input: [Batch, 1, 48, 48] (must be divisible by scale 3 for perfect reconstruction)
    x = torch.randn(1, 1, 48, 48)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Check if gradients flow
    loss = y.sum()
    loss.backward()
    print("Backward pass successful")
