import torch
import torch.nn as nn

class LayerNormFunction(torch.autograd.Function):
    """
    自定义2D层归一化自动求导函数（比PyTorch原生LayerNorm更适配通道优先的图像特征）
    功能：对通道维度（dim=1）进行归一化，支持自定义权重和偏置，手动实现前向/反向传播
    """
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        """
        前向传播：计算通道维度的均值和方差 → 归一化 → 权重缩放+偏置
        Args:
            ctx: 上下文对象，用于保存反向传播所需参数
            x: 输入特征，shape=(N, C, H, W)（N=批量，C=通道，H=高，W=宽）
            weight: 通道缩放权重，shape=(C,)
            bias: 通道偏置，shape=(C,)
            eps: 防止除零的小值（默认1e-6）
        Returns:
            y: 归一化后特征，shape=(N, C, H, W)
        """
        ctx.eps = eps
        N, C, H, W = x.size()
        # 通道维度均值（keepdim=True保留通道维度，便于广播）
        mu = x.mean(1, keepdim=True)  # shape=(N, 1, H, W)
        # 通道维度方差
        var = (x - mu).pow(2).mean(1, keepdim=True)  # shape=(N, 1, H, W)
        # 归一化：(x - 均值) / sqrt(方差 + eps)
        y = (x - mu) / (var + eps).sqrt()
        # 保存反向传播所需参数
        ctx.save_for_backward(y, var, weight)
        # 权重缩放+偏置（权重和偏置需reshape为(1,C,1,1)适配特征维度）
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：手动计算输入x、权重weight、偏置bias的梯度
        Args:
            ctx: 前向传播保存的上下文
            grad_output: 上层传入的梯度，shape=(N, C, H, W)
        Returns:
            gx: 输入x的梯度，shape=(N, C, H, W)
            grad_weight: 权重weight的梯度，shape=(C,)
            grad_bias: 偏置bias的梯度，shape=(C,)
            None: eps无梯度（无需优化）
        """
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        # 读取前向传播保存的参数
        y, var, weight = ctx.saved_variables
        # 计算权重缩放后的梯度
        g = grad_output * weight.view(1, C, 1, 1)  # shape=(N, C, H, W)
        # 梯度的通道均值
        mean_g = g.mean(dim=1, keepdim=True)  # shape=(N, 1, H, W)
        # 梯度与归一化后特征y的通道均值
        mean_gy = (g * y).mean(dim=1, keepdim=True)  # shape=(N, 1, H, W)
        # 输入x的梯度计算（基于层归一化反向公式推导）
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        # 权重的梯度（梯度输出与y的空间求和）
        grad_weight = (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0)  # shape=(C,)
        # 偏置的梯度（梯度输出的空间求和）
        grad_bias = grad_output.sum(dim=3).sum(dim=2).sum(dim=0)  # shape=(C,)
        return gx, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    """
    2D层归一化模块（基于自定义LayerNormFunction）
    适配通道优先的图像特征（N,C,H,W），替代PyTorch原生LayerNorm（默认最后一维归一化）
    """
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        
        # 注册可训练参数：通道缩放权重（初始1）和偏置（初始0）
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps  # 防止除零的小值

    def forward(self, x):
        """前向传播：调用自定义LayerNormFunction"""
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """
    简单门控机制（通道拆分+元素乘）
    功能：将输入通道拆分为两部分，通过元素乘实现特征筛选，增强模型非线性表达
    """
    def forward(self, x):
        # 沿通道维度（dim=1）拆分为两等份（x1和x2通道数均为原通道数的1/2）
        x1, x2 = x.chunk(2, dim=1)
        # 元素乘：筛选有效特征（抑制噪声，保留关键信息）
        return x1 * x2


class FreMLP(nn.Module):
    """
    频率域MLP模块（FreMLP）
    功能：在频率域对特征进行处理，捕捉传统空间域难以提取的频率信息（如周期性纹理、全局亮度变化）
    核心流程：空间域→频率域（FFT）→幅度处理→频率域→空间域（IFFT）
    """
    def __init__(self, nc, expand=2):
        """
        Args:
            nc: 输入/输出通道数（频率域处理不改变通道数）
            expand: MLP中间层通道扩展倍数（默认2，增强频率特征表达）
        """
        super(FreMLP, self).__init__()
        
        # 频率域幅度处理模块（1×1卷积实现通道维度变换与特征优化）
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),  # 升维：nc → expand×nc
            nn.LeakyReLU(0.1, inplace=True),       # 非线性激活（LeakyReLU避免梯度消失）
            nn.Conv2d(expand * nc, nc, 1, 1, 0)    # 降维：expand×nc → nc（恢复原通道数）
        )

    def forward(self, x):
        """
        前向传播：空间域→频率域→幅度处理→相位恢复→空间域
        Args:
            x: 输入空间域特征，shape=(N, nc, H, W)
        Returns:
            x_out: 输出空间域特征，shape=(N, nc, H, W)（与输入维度一致）
        """
        _, _, H, W = x.shape
        
        # 1. 空间域→频率域：二维实值快速傅里叶变换（rfft2，输出复数特征）
        # norm='backward'确保FFT与IFFT互为逆运算（无能量损失）
        x_freq = torch.fft.rfft2(x, norm='backward')  # shape=(N, nc, H, W//2+1)（复数）
        
        # 2. 分离频率域的幅度（mag）和相位（pha）
        mag = torch.abs(x_freq)    # 幅度：反映频率成分的强度，shape=(N, nc, H, W//2+1)
        pha = torch.angle(x_freq)  # 相位：反映频率成分的位置，shape=(N, nc, H, W//2+1)
        
        # 3. MLP处理幅度（相位保持不变，避免破坏空间位置信息）
        mag = self.process1(mag)   # 幅度优化：增强有用频率，抑制噪声频率
        
        # 4. 从幅度和相位恢复复数特征
        real = mag * torch.cos(pha)  # 实部：幅度×cos(相位)
        imag = mag * torch.sin(pha)  # 虚部：幅度×sin(相位)
        x_out = torch.complex(real, imag)  # 重组复数特征，shape=(N, nc, H, W//2+1)
        
        # 5. 频率域→空间域：二维逆快速傅里叶变换（irfft2，恢复实值特征）
        # s=(H,W)指定输出空间尺寸，确保与输入一致
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')  # shape=(N, nc, H, W)
        return x_out


class Branch(nn.Module):
    """
    扩张卷积分支模块（EBlock的空间特征提取单元）
    功能：通过不同 dilation 率的深度卷积，捕捉多尺度空间特征（局部细节+长程依赖）
    """
    def __init__(self, c, DW_Expand, dilation=1):
        """
        Args:
            c: 输入特征的基础通道数
            DW_Expand: 深度卷积通道扩展倍数（默认2，增强通道表达）
            dilation: 扩张率（默认1，dilation>1时实现大感受野）
        """
        super().__init__()
        self.dw_channel = DW_Expand * c  # 深度卷积的通道数（扩展后）
        
        # 深度卷积（groups=dw_channel，每个通道独立卷积，降低计算量）
        self.branch = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dw_channel,
                out_channels=self.dw_channel,
                kernel_size=3,
                padding=dilation,  # 适配扩张率的padding（确保输入输出尺寸一致）
                stride=1,
                groups=self.dw_channel,  # 深度卷积：分组数=通道数
                bias=True,
                dilation=dilation  # 扩张率（控制感受野大小）
            )
        )

    def forward(self, input):
        """前向传播：输入特征→深度卷积→输出（尺寸不变）"""
        return self.branch(input)


class EBlock(nn.Module):
    """
    核心模块：EBlock（融合空间多尺度特征与频率域特征的双阶段特征增强模块）
    第一阶段：空间域多分支扩张卷积+门控+注意力；第二阶段：频率域FreMLP+特征融合
    """
    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        """
        Args:
            c: 输入/输出通道数（模块保持通道数不变）
            DW_Expand: 深度卷积通道扩展倍数（默认2）
            dilations: 扩张卷积分支的扩张率列表（默认[1]，支持多分支多尺度）
            extra_depth_wise: 是否添加额外深度卷积（默认False，可选增强空间细节）
        """
        
        super().__init__()
        self.dw_channel = DW_Expand * c  # 深度卷积的扩展通道数
        
        # 可选额外深度卷积（增强空间局部细节，默认关闭）
        self.extra_conv = nn.Conv2d(
            c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1
        ) if extra_depth_wise else nn.Identity()
        
        # 1×1卷积：空间特征通道升维（c→dw_channel，为深度卷积做准备）
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=self.dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
            dilation=1
        )
        
        # 多扩张卷积分支（根据dilations列表创建，捕捉多尺度空间特征）
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=dilation))
       
        # 校验：扩张率列表长度与分支数量一致
        assert len(dilations) == len(self.branches)
        
        # 空间注意力模块（SCA：Spatial Channel Attention）：全局平均池化+1×1卷积
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化：(N, dw_channel//2, H, W)→(N, dw_channel//2, 1, 1)
            nn.Conv2d(
                in_channels=self.dw_channel // 2,
                out_channels=self.dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
                dilation=1
            )  # 通道注意力：生成通道权重
        )
        
        self.sg1 = SimpleGate()  # 门控机制：筛选空间特征
        
        # 1×1卷积：空间特征通道降维（dw_channel//2→c，恢复原通道数）
        self.conv3 = nn.Conv2d(
            in_channels=self.dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
            dilation=1
        )
        
        # 第二阶段：归一化与频率域处理
        self.norm1 = LayerNorm2d(c)  # 第一阶段输入归一化
        self.norm2 = LayerNorm2d(c)  # 第二阶段输入归一化
        self.freq = FreMLP(nc=c, expand=2)  # 频率域处理模块
        
        # 特征融合参数（可训练，控制频率特征的贡献度）
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)  # 频率特征权重
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)   # 空间特征权重

    def forward(self, inp):
        """
        前向传播：双阶段特征增强（空间阶段→频率阶段）
        Args:
            inp: 输入特征，shape=(N, c, H, W)
        Returns:
            x: 输出特征，shape=(N, c, H, W)（与输入维度一致）
        """
        
        # 保存输入用于残差连接
        y = inp
        
        # -------------------------- 第一阶段：空间域多尺度特征增强 --------------------------
        # 归一化→（可选）额外深度卷积→通道升维
        x = self.norm1(inp)  # (N,c,H,W)→(N,c,H,W)（通道归一化）
        x = self.conv1(self.extra_conv(x))  # (N,c,H,W)→(N,dw_channel,H,W)（升维）
        
        # 多扩张卷积分支求和（多尺度空间特征融合）
        z = 0
        for branch in self.branches:
            z += branch(x)  # 每个分支输出均为(N,dw_channel,H,W)，求和后维度不变
       
        # 门控筛选→空间注意力加权→通道降维
        z = self.sg1(z)  # (N,dw_channel,H,W)→(N,dw_channel//2,H,W)（通道拆分+元素乘）
        x = self.sca(z) * z  # 注意力加权：(N,dw_channel//2,1,1) * (N,dw_channel//2,H,W)→(N,dw_channel//2,H,W)
        x = self.conv3(x)  # (N,dw_channel//2,H,W)→(N,c,H,W)（降维，恢复原通道数）
        
        # 空间特征与输入残差融合（beta控制空间特征贡献度）
        y = inp + self.beta * x  # (N,c,H,W) + (N,c,1,1)*(N,c,H,W)→(N,c,H,W)

        # -------------------------- 第二阶段：频率域特征增强与融合 --------------------------
        
        # 归一化→频率域处理→特征融合
        x_step2 = self.norm2(y)  # (N,c,H,W)→(N,c,H,W)（归一化，稳定频率域处理）
        x_freq = self.freq(x_step2)  # (N,c,H,W)→(N,c,H,W)（频率域特征优化）
        
        # 频率特征与空间特征加权融合→残差连接（gamma控制频率特征贡献度）
        x = y * x_freq  # 频率特征加权空间特征：(N,c,H,W)*(N,c,H,W)→(N,c,H,W)
        x = y + x * self.gamma  # 最终残差融合：(N,c,H,W) + (N,c,1,1)*(N,c,H,W)→(N,c,H,W)

        return x

# if __name__ == "__main__":
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#     x = torch.randn(1, 64, 32, 32).to(device)
#
#     model = EBlock(64)
#
#     model.to(device)
#     y = model(x)
#
#     print("微信公众号：十小大的底层视觉工坊")
#     print("知乎、CSDN：十小大")
#
#     print("输入特征维度：", x.shape)
#     print("输出特征维度：", y.shape)
