import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
try:
    from mamba_ssm import Mamba
except:
    pass
class PVMamba(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2) #将B C L -->B L C
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out.permute(0,2,3,1)

class FeedForward(nn.Module):
    """
    MLP block with pre-layernorm, GELU activation, and dropout.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MPMA(nn.Module):  #B C H W Multipole Attention 多极注意力
    """
    Hierarchical local attention across multiple scales with down/up-sampling.
    """

    def __init__(
        self,
        in_channels,
        image_size,
        local_attention_kernel_size=2,
        local_attention_stride=2,
        downsampling="conv",
        upsampling= "conv",
        sampling_rate=2,
        heads=4,
        dim_head=16,
        dropout=0.1,
        channel_scale=1,
    ):
        super().__init__()

        # self.levels = int(math.log(image_size, sampling_rate))  # math.log(x, base)
        self.levels = 5
        channels_conv = [in_channels * (channel_scale**i) for i in range(self.levels)]

        # A shared local attention layer for all levels
        # self.Attention = LocalAttention2D(
        #     kernel_size=local_attention_kernel_size,
        #     stride=local_attention_stride,
        #     dim=channels_conv[0],
        #     heads=heads,
        #     dim_head=dim_head,
        #     dropout=dropout,
        # )
        self.MambaAttention = PVMamba(in_channels,in_channels)

        if downsampling == "avg_pool":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate),
                Rearrange("B C H W -> B H W C"),
            )

        elif downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

        if upsampling == "avg_pool":
            current = image_size

            for _ in range(self.levels):
                assert (
                    current % sampling_rate == 0
                ), f"Image size not divisible by sampling_rate size at level {_}: current={current}, sampling_ratel={sampling_rate}"
                current = current // sampling_rate

            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),
                Rearrange("B C H W -> B H W C"),
            )

        elif upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

    def forward(self, x):
        # x: [B, H, W, C], returns the same shape
        # Level 0
        x = x.permute(0,2,3,1) #要求是B H W C ,所以调整x的B C H W的维度顺序
        x_in = x

        x_out = []
        # x_out.append(self.Attention(x_in))
        x_out.append(self.MambaAttention(x_in))
        # Levels from 1 to L
        for l in range(1, self.levels):
            x_in = self.down(x_in)
            x_out_down = self.MambaAttention(x_in)
            x_out.append(x_out_down)

        res = x_out.pop()
        for l, out_down in enumerate(x_out[::-1]):
            res = out_down + (1 / (l + 1)) * self.up(res)

        return res.permute(0,3,2,1)

class MPMABlock(nn.Module): # MultipoleMambaAttention多极曼巴注意力
    """
    Transformer block stacking multiple Multipole_Attention2D + FeedForward layers.
    """

    def __init__(
        self,
        in_channels,
        image_size,
        kernel_size=2,  # Local attention patch size
        local_attention_stride=2,  # stride（与 kernel_size 相同）
        downsampling="conv",  # 使用卷积做下采样
        upsampling="conv",  # 使用反卷积做上采样
        sampling_rate=2,  # 每层下采样/上采样缩放因子
        depth=2,  # 堆叠层数
        heads=4,  # 注意力头数
        dim_head=16,  # 每个头的维度
        att_dropout=0.1,  # 注意力 dropout
        channel_scale=1,  # 多尺度通道扩展倍率（设为1保持通道数一致）

    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.layers = nn.ModuleList([])
        mlp_dim = int(4*in_channels)  # FeedForward中间层维度
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        MPMA(
                            in_channels,
                            image_size,
                            kernel_size,
                            local_attention_stride,
                            downsampling,
                            upsampling,
                            sampling_rate,
                            heads,
                            dim_head,
                            att_dropout,
                            channel_scale,
                        ),
                        FeedForward(in_channels, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        """
        Expected input shape: [B, H, W, C]
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x.permute(0,2,3,1)).permute(0,3,1,2) + x
        return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if device.type == 'cpu':
#         print("need CUDA support")
#         exit()
#
#     input_tensor = torch.randn(1, 32, 64, 64).to(device)
#
#     mpma_module = MPMA(in_channels=32, image_size=64).to(device)
#     output1 = mpma_module(input_tensor)
#     print('MPMA_input_size:', input_tensor.size())
#     print('MPMA_output_size:', output1.size())
#     print("-" * 30)
#
#     mpma_block_module = MPMABlock(in_channels=32, image_size=64).to(device)
#     output2 = mpma_block_module(input_tensor)
#     print('MPMABlock_input_size:', input_tensor.size())
#     print('MPMABlock_output_size:', output2.size())
