import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)
        self.norm_type = norm_type
        self.act = activation
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        elif self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        else:
            self.norm = None

        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class AdaptiveCombiner(nn.Module):
    def __init__(self):
        super(AdaptiveCombiner, self).__init__()
        self.d = nn.Parameter(torch.randn(1, 1, 1, 1))  # 可学习标量

    def forward(self, p, i):
        batch_size, channel, w, h = p.shape
        d = self.d.expand(batch_size, channel, w, h)
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        channel_att = self.fc(avg_out).view(x.size(0), x.size(1), 1, 1)
        return x * channel_att

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.conv(x)
        att = self.sigmoid(att)
        return x * att

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AFEM(nn.Module):
    def __init__(self, in_channels, out_channels, chunk_count=4):
        super(AFEM, self).__init__()
        self.chunk_count = chunk_count
        self.adaptive_combiner = AdaptiveCombiner()
        self.cbam = CBAM(in_channels // chunk_count)

        self.tail_conv = nn.Sequential(
            conv_block(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))
        )

        self.refine = nn.Sequential(
            conv_block(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x_low, x_high):
        if x_high is None or x_low is None:
            raise ValueError("x_high and x_low cannot be None")

        x_high_chunks = torch.chunk(x_high, self.chunk_count, dim=1)
        x_low = F.interpolate(x_low, size=x_high_chunks[0].shape[2:], mode='bilinear', align_corners=True)
        x_low_chunks = torch.chunk(x_low, self.chunk_count, dim=1)

        fused = []
        for xl, xh in zip(x_low_chunks, x_high_chunks):
            mix = self.adaptive_combiner(xl, xh)  # 自适应融合
            mix = self.cbam(mix)                  # 注意力增强
            fused.append(mix)

        x = torch.cat(fused, dim=1)
        x = self.tail_conv(x)
        x = self.refine(x) + x  # 残差连接
        return x


class FinalFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels=3, reduction_ratio=4):
        super().__init__()
        self.conv_in = conv_block(in_channels, in_channels, norm_type='bn')

        self.cbam = CBAM(in_channels, reduction_ratio)

        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, detail, lf_noise, stripe_noise):
        x = torch.cat([detail, lf_noise, stripe_noise], dim=1)

        x = self.conv_in(x)
        x_att = self.cbam(x)

        x = x + x_att

        output = self.conv_out(x)
        return output

# # in NAFMamba __init__
# self.fusion_head = FinalFusionModule(in_channels=img_channel * 3, out_channels=img_channel)
#
# # at the end of forward
# final_clean_image = self.fusion_head(pred_detail, pred_lf_noise, pred_stripe_noise)
