import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import to_2tuple
from torch.nn import functional as F
# DynamicFilter来自：https://github.com/okojoalg/dfformer/blob/main/models/dfformer.py
# DynamicFilter来自论文： https://arxiv.org/pdf/2303.03932v2
#

#FDSM 来自CVPR 2025
#FDSM 论文地址;https://arxiv.org/abs/2412.16645
#看Ai缝合怪b站视频：2025.6.11更新的视频
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias
def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    new_weight = torch.nn.functional.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight
class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class FrequencyDynamicSelection(nn.Module): #原论文名字：DynamicFilter
    def __init__(self, dim, expansion_ratio=1, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=64, weight_resize=True,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x,dw):
        B, H, W, _ = x.shape


        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,-1).softmax(dim=1)

        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],
                                                    x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)

        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)
        return x

class FDSM(nn.Module):
    def __init__(self, in_channels,size):
        super(FDSM, self).__init__()
        # RGB and NIR feature extraction
        self.rgb_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.nir_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.cpcs = nn.Sequential(
                            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
                            nn.PReLU(),
                            nn.Conv2d(in_channels, in_channels , kernel_size=3, padding=1),
                            nn.SiLU())

        # Aggregation layers
        self.mlp = ConvMlp(in_channels )
        self.softmax = nn.Softmax(dim=1)

        self.fds =FrequencyDynamicSelection(in_channels,size= size)

    def forward(self, rgb, nir):
        b,c,h,w = rgb.shape
        rgb_feat = self.rgb_conv(rgb)
        nir_feat = self.nir_conv(nir)

        frn = self.cpcs(torch.cat([rgb_feat,nir_feat],dim=1))
        att = F.adaptive_avg_pool2d(frn, output_size=h)  # B, C, h,w
        Aggregation = self.softmax(self.mlp(att))

        rgb_feat =rgb_feat.permute(0,2,3,1)
        nir_feat = nir_feat.permute(0,2,3,1)
        Aggregation = Aggregation.permute(0,2,3,1)

        feat_r = self.fds(rgb_feat, Aggregation)
        feat_n = self.fds(nir_feat,Aggregation)
        # return feat_r, feat_n
        return feat_r+feat_n

#二次创新FDAM，频率动态注意力混合模块，冲SCI三区和四区，CCF-B/C

'''
#论文地址： https://arxiv.org/pdf/2403.10067
论文题目：用于高光谱图像去噪的混合卷积和注意力网络     IEEE 2024
卷积和注意力特征融合模块：CAFM        
背景：  
摘要—高光谱图像（HSI）去噪对于高光谱数据的有效分析和解释至关重要。
然而，很少有人探索同时对全局和局部特征进行建模来增强 HSI 去噪。
在这篇文章中，我们提出了一种混合卷积和注意力网络（HCANet），它利用了卷积神经网络（CNNs）和Transformers的优势。
为了增强全局和局部特征的建模，我们设计了一种卷积和注意力融合模块，旨在捕获长程依赖性和邻域光谱相关性。

CAFM:所提出的卷积和注意力特征融合模块。它由本地和全局分支机构组成。
在局部分支中，采用卷积和通道洗牌进行局部特征提取。
在全局分支中，注意力机制用于对长程特征依赖关系进行建模。

适用于：高光谱图像去噪，图像增强，图像分类，目标检测，图像分割，暗光增强等所有CV2D任务
'''
class CAFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True, groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)
        out_conv = out_conv.squeeze(2)

        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv
        return output
class FDAM(nn.Module):
    def __init__(self, in_channels,size):
        super(FDAM, self).__init__()
        # RGB and NIR feature extraction
        self.rgb_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.nir_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.nir_rgb_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.CAFM = CAFM(in_channels)
        #在此处利用CAFM模块替换原来的普通卷积和激活函数，
        # 对融合后的特征图在局部分支中，采用卷积和通道洗牌进行局部特征提取。
        # 对融合后的特征图在全局分支中，注意力机制用于对长程特征依赖关系进行建模。
        # CAFM这个模块很通用，既考虑局部特征提取有考虑长距离全局特征捕捉，

        # Aggregation layers
        self.mlp = ConvMlp(in_channels )
        self.softmax = nn.Softmax(dim=1)

        self.fds =FrequencyDynamicSelection(in_channels,size=size)

    def forward(self, rgb, nir):
        b,c,h,w = rgb.shape
        rgb_feat = self.rgb_conv(rgb)
        nir_feat = self.nir_conv(nir)

        frn = self.CAFM(self.nir_rgb_conv(torch.cat([rgb_feat,nir_feat],dim=1)))
        att = F.adaptive_avg_pool2d(frn, output_size=h)  # B, C, h,w
        Aggregation = self.softmax(self.mlp(att))

        rgb_feat =rgb_feat.permute(0,2,3,1)
        nir_feat = nir_feat.permute(0,2,3,1)
        Aggregation = Aggregation.permute(0,2,3,1)

        feat_r = self.fds(rgb_feat, Aggregation)
        feat_n = self.fds(nir_feat,Aggregation)
        # return feat_r.permute(0,3,1,2), feat_n.permute(0,3,1,2)
        return feat_r.permute(0,3,1,2)+feat_n.permute(0,3,1,2)


# # 输入 B C H W,  输出 B C H W
# if __name__ == '__main__':
#     # 定义输入张量的形状为 B, C, H, W
#     input1= torch.randn(1, 32, 64, 64)
#     input2 = torch.randn(1, 32, 64, 64)
#     # 创建 FDSM 模块
#     fdsm = FDSM(in_channels=32,size =64) #size代表输入特征图H或是W
#     # 将输入图像传入FDSM 模块进行处理
#     output = fdsm(input1,input2).permute(0,3,1,2)
#     # 输出结果的形状
#     # 打印输入和输出的形状
#     print('Ai缝合即插即用模块永久更新-FDSM_input_size:', input1.size())
#     print('Ai缝合即插即用模块永久更新-FDSM_output_size:', output.size())
#
#     # CVPR2025 FDSM模块的二次创新，FDAM在我的二次创新模块交流群，可以直接去发小论文！
#     # 创建 FDAM 模块
#     fdam = FDAM(in_channels=32,size=64)#size代表输入特征图H或是W
#     # 将输入图像传入FDAM 模块进行处理
#     output = fdam(input1,input2)
#     print('顶会顶刊二次创新模块永久更新-FDAM_input_size:', input1.size())
#     print('顶会顶刊二次创新模块永久更新-FDAM_output_size:', output.size())
