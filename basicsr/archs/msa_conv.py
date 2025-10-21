import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
#来自B站：Ai缝合怪整理的即插即用模块
##看Ai缝合怪b站视频：2025.8.21更新的视频
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
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
#考虑小伙伴的需要，所以把原模块也放进来了
class MRFAConv(nn.Module):                                                                                                                                                                                   # Ai缝合怪整理的即插即用模块
    def __init__(self, dim):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a1 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 4, 7, padding=3, groups=dim // 4)
        )

        self.v1 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim//4)

        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.a2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 9, padding=4, groups=dim // 2)
        )

        self.v2 = nn.Conv2d(dim//2, dim//2, 1)
        self.v21 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v22 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj2 = nn.Conv2d(dim // 2, dim // 4, 1)
        self.conv3_2 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")
        self.a3 = nn.Sequential(
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 11, padding=5, groups=dim * 3 // 4)
        )
        self.v3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v31 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v32 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj3 = nn.Conv2d(dim * 3 // 4, dim // 4, 1)
        self.conv3_3 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.dim = dim

    def forward(self, x):                                                                                                                                                                               #Ai缝合怪整理的即插即用模块

        x = self.norm1(x)
        x_split = torch.split(x, self.dim // 4, dim=1)
        a = self.a1(x_split[0])
        mul = a * self.v1(x_split[0])
        mul = self.v11(mul)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a
        x1 = torch.cat((x1, mul), dim=1)

        x1 = self.norm2(x1)
        a = self.a2(x1)
        mul = a * self.v2(x1)
        mul = self.v21(mul)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a)
        x2 = torch.cat((x2, mul), dim=1)

        x2 = self.norm3(x2)
        a = self.a3(x2)
        mul = a * self.v3(x2)
        mul = self.v31(mul)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a)
        x = torch.cat((x3, mul), dim=1)
        return x
class ConvAtt(nn.Module):
    def __init__(self, in_channels, att_channels=16, lk_size=13, sk_size=3, reduction=2):
        """
        :param in_channels: AIFHG输入特征图通道数
        :param att_channels: AIFHG用于注意力通道数，默认为16
        :param lk_size: AIFHG 静态大核卷积核尺寸（如图中13）
        :param sk_size: AIFHG动态卷积核尺寸（如图中3）
        :param reduction: AIFHG动态卷积中间层压缩因子
        """
        super().__init__()
        self.in_channels = in_channels
        self.att_channels = att_channels
        self.idt_channels = in_channels - att_channels
        self.lk_size = lk_size
        self.sk_size = sk_size

        # 动态卷积核生成器
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(att_channels, att_channels // reduction, 1),
            nn.GELU(),#AIFHG
            nn.Conv2d(att_channels // reduction, att_channels * sk_size * sk_size, 1)
        )
        nn.init.zeros_(self.kernel_gen[-1].weight)
        nn.init.zeros_(self.kernel_gen[-1].bias)

        # 共享静态大核卷积核：定义为参数，非卷积层
        self.lk_filter = nn.Parameter(torch.randn(att_channels, att_channels, lk_size, lk_size))
        nn.init.kaiming_normal_(self.lk_filter, mode='fan_out', nonlinearity='relu')

        # 融合层
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.att_channels + self.idt_channels, f"Input channel {C} must match att + idt ({self.att_channels} + {self.idt_channels})"

        # 通道拆分
        F_att, F_idt = torch.split(x, [self.att_channels, self.idt_channels], dim=1)

        # 生成动态卷积核 [B * att, 1, 3, 3]
        kernel = self.kernel_gen(F_att).reshape(B * self.att_channels, 1, self.sk_size, self.sk_size)

        # 动态卷积操作
        F_att_re = rearrange(F_att, 'b c h w -> 1 (b c) h w')
        out_dk = F.conv2d(F_att_re, kernel, padding=self.sk_size // 2, groups=B * self.att_channels)
        out_dk = rearrange(out_dk, '1 (b c) h w -> b c h w', b=B, c=self.att_channels)

        # 静态大核卷积
        out_lk = F.conv2d(F_att, self.lk_filter, padding=self.lk_size // 2)

        # 融合（两个卷积结果加和）
        out_att = out_lk + out_dk

        # 拼接 F_idt（保留通道）
        out = torch.cat([out_att, F_idt], dim=1)

        # 1x1 融合
        out = self.fusion(out)
        return out
# MSAConv多尺度注意力卷积模块 是ICCV2025 MRFAConv（提示一下：原论文没有这个模块名，是我帮助小伙伴理解它的三层RFA，取得这个名字） 的二次创新模块，
class MSAConv(nn.Module): #二次创新这个模块名也可以取：MRAConv多感受野注意力卷积模块; MRCAB多感受野卷积注意力块 ；MSCAB多尺度卷积注意力块等都是独一无二多；模块名小伙伴可以灵活去定义它                                                                                                                                                                                  # Ai缝合怪整理的即插即用模块
    def __init__(self, dim,lk_size=[7,9,11]): #lk_size这个大核列表自己也可以取其它的组，多去做一下实验和消融实验：比如[5,7,9]，[7,9,11]，[9,11,13] 等等
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.a1 = ConvAtt(in_channels=dim // 4, lk_size=lk_size[0])

        self.v1 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim//4)

        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.a2 = ConvAtt(in_channels=dim // 2, lk_size=lk_size[1])

        self.v2 = nn.Conv2d(dim//2, dim//2, 1)
        self.v21 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v22 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj2 = nn.Conv2d(dim // 2, dim // 4, 1)
        self.conv3_2 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")

        self.a3 = ConvAtt(in_channels=dim * 3 // 4, lk_size=lk_size[2])

        self.v3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v31 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v32 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj3 = nn.Conv2d(dim * 3 // 4, dim // 4, 1)
        self.conv3_3 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.dim = dim

    def forward(self, x):                                                                                                                                                                               #Ai缝合怪整理的即插即用模块

        x = self.norm1(x)
        x_split = torch.split(x, self.dim // 4, dim=1)
        a = self.a1(x_split[0])
        mul = a * self.v1(x_split[0])
        mul = self.v11(mul)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a
        x1 = torch.cat((x1, mul), dim=1)

        x1 = self.norm2(x1)
        a = self.a2(x1)
        mul = a * self.v2(x1)
        mul = self.v21(mul)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a)
        x2 = torch.cat((x2, mul), dim=1)

        x2 = self.norm3(x2)
        a = self.a3(x2)
        mul = a * self.v3(x2)
        mul = self.v31(mul)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a)
        x = torch.cat((x3, mul), dim=1)
        return x
# MSAConv多尺度注意力卷积模块 是ICCV2025 MRFAConv的二次创新
#谈一下我为什么给小伙伴带来这个MSAConv二次创新模块的原因，我提到的每一个点都适合小伙伴去疯狂结合自己的任务去编故事！轻松去发SCI二区/三区/四区，或是CCF-B/C
'''
第一点：首先这个模块架构设计也比较独特，首先将输入特征图按照通道维度划分成多个子部分，包括一个主干通道和多个辅助通道，每一组通道会被送入对应的子模块处理。 好编故事！
      （我们通常的见到最多的，就是通过多尺度采用不同大核空洞卷积去提取特征，然后再进行残差连接或是通道拼接操作去处理）
第二点：多层感受野叠加策略/层级式操作策略，这一点大家不要感到陌生，顶会顶刊论文都在用。 好编故事！
       举例：今年AAAI2025 FBRT-YOLO中多MKP模块也是采用这种多层感受野叠加思想。AAAI2025 FBRT-YOLO论文地址：https://arxiv.org/abs/2504.20670
第三点：感受野分布控制，通过中心主干通道逐步扩展并融合周围小尺度通道的细节，使感受野的响应分布更接近高斯形态（即越靠近输出位置，特征的权重越高，越远则影响越小）。好编故事！

第四点：受到自注意力机制在层间重复性的启发，ConvAttn 采用一个共享的大卷积核模拟长距离建模能力，
       用每层输入自适应的动态卷积核模拟自注意力的加权机制，在大幅减少内存和计算代价的同时，保持了Transformer级别的表征能力。好编故事！

'''
# # 输入 B C H W,  输出 B C H W
# if __name__ == '__main__':
#     # 定义输入张量的形状为 B, C, H, W
#     input= torch.randn(1, 64, 64, 64)
#     # 创建 MSAConv模块
#     MSAConv = MSAConv(dim=64,lk_size=[7,9,11])
#     # 将输入图像传入MSAConv 模块进行处理
#     output = MSAConv(input)
#     # 打印输入和输出的形状
#     print('Ai缝合怪二次创新模块永久更新-多尺度注意力卷积_MSAConv_input_size:', input.size())
#     print('Ai缝合怪二次创新模块永久更新-多尺度注意力卷积_MSAConv_output_size:', output.size())
