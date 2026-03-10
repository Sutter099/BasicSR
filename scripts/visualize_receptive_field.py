import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# # ---------------------------------------------------------
# # 1. 核心可视化函数
# # ---------------------------------------------------------
# def visualize_erf(model, input_size=(1, 1, 256, 256), device='cuda', save_path=None):
#     model.to(device)
#     model.eval()
#
#     # ERF 使用全零输入来纯粹地计算感受野梯度
#     input_img = torch.zeros(input_size).to(device).requires_grad_(True)
#
#     output = model(input_img)
#     _, _, h, w = output.shape
#     central_pixel = output[:, :, h//2, w//2].sum()
#
#     model.zero_grad()
#     central_pixel.backward()
#
#     # grad = input_img.grad.detach().cpu().numpy().squeeze()
#     # grad = np.abs(grad)
#     # grad = np.log1p(grad) # 对数缩放凸显微弱的远距离梯度
#
#     grad = input_img.grad.detach().cpu().numpy().squeeze() # 此时 shape 为 (3, 256, 256) 或 (256, 256)
#     grad = np.abs(grad)
#     if grad.ndim == 3:
#         grad = np.sum(grad, axis=0)
#
#
#
#
#
#
#     # ==========================================
#     # 🌟 核心数据处理修改
#     # ==========================================
#
#     # 1. 消除边界效应 (Boundary Artifacts)
#     # 强行将图像边缘 5 个像素的异常梯度抹零，防止它们破坏全局归一化
#     margin = 5
#     grad[:margin, :] = 0
#     grad[-margin:, :] = 0
#     grad[:, :margin] = 0
#     grad[:, -margin:] = 0
#
#     # 2. 更强效的对数拉伸
#     # 使用 log10(x + 1e-5) 对微弱远距离梯度的放大效果比 log1p 更好，常用于 ERF 论文
#     grad = np.log10(grad + 1e-5)
#
#     # 3. 稳健归一化 (Robust Normalization)
#     # 不要使用原生的 .max()，而是用 99.8% 的分位数，截断掉偶尔出现的单个极高亮噪点
#     grad = grad - grad.min()
#     vmax = np.percentile(grad, 99.8) 
#     grad = np.clip(grad, 0, vmax)      # 大于 vmax 的值强制等于 vmax
#     grad = grad / (vmax + 1e-8)        # 缩放到 [0, 1]
#
#     # ==========================================
#
#
#
#
#
#
#
#
#
#
#
#
#     # grad = np.log1p(grad)
#     #
#     # # 归一化
#     # grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
#
#     plt.figure(figsize=(6, 5))
#     im = plt.imshow(grad, cmap='YlGn', vmin=0, vmax=1)
#     plt.axis('off')
#     cbar = plt.colorbar(im, shrink=0.8, aspect=20, pad=0.05)
#     cbar.ax.tick_params(labelsize=8)
#     # plt.colorbar()
#     plt.title("Effective Receptive Field (ERF)")
#
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300) # 提高 dpi 满足论文要求
#         print(f"ERF image saved to {save_path}")
#     else:
#         plt.show()

def visualize_erf(model, input_size=(1, 3, 256, 256), device='cuda', save_path=None, num_samples=200):
    model.to(device)
    model.eval()
    
    _, _, h, w = input_size
    accumulated_grad = np.zeros((h, w))
    
    print(f"Calculating ERF over {num_samples} random samples...")
    
    # 🌟 关键修正 1：循环累加多个随机噪声图像的梯度，消除单次随机带来的噪点
    for i in range(num_samples):
        # 使用随机噪声 (0~1均匀分布) 代替全 0 输入，打破 Transformer 的对称性
        input_img = torch.rand(input_size).to(device).requires_grad_(True)
        
        output = model(input_img)
        # 取输出中心点
        central_pixel = output[:, :, h//2, w//2].sum()
        
        model.zero_grad()
        central_pixel.backward()
        
        grad = input_img.grad.detach().cpu().numpy().squeeze()
        grad = np.abs(grad)
        
        if grad.ndim == 3:
            grad = np.mean(grad, axis=0)
            
        accumulated_grad += grad

    # 取平均梯度
    grad = accumulated_grad / num_samples

    # 🌟 关键修正 2：加宽边界清理范围，彻底干掉 Padding 效应
    margin = 10
    grad[:margin, :] = 0
    grad[-margin:, :] = 0
    grad[:, :margin] = 0
    grad[:, -margin:] = 0

    # 🌟 关键修正 3：回归最经典的 Log 缩放 + Min-Max 归一化
    # 放弃 percentile，因为多次平均后的梯度分布已经非常健康了
    grad = np.log10(grad + 1e-7)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

    # 绘图部分
    plt.figure(figsize=(5, 5))
    im = plt.imshow(grad, cmap='YlGn', vmin=0, vmax=1)
    plt.axis('off')
    
    cbar = plt.colorbar(im, shrink=0.8, aspect=20, pad=0.05)
    cbar.ax.tick_params(labelsize=8)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
        print(f"ERF image saved to {save_path}")
    else:
        plt.show()

def visualize_lam(model, img_path, target_patch, steps=20, device='cuda', save_path=None):
    # 读取红外图像（单通道灰度图）并预处理
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)




    # INFO: to reduce GPU memory consumption
    # 在 cv2 读取图像后，加上裁剪逻辑
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    if img is None:
        raise FileNotFoundError(f"Cannot load image at {img_path}")
    
    # 假设你的 target 坐标是 (128, 128)，我们可以裁剪出一个 256x256 的区域
    # 注意要确保裁剪边界不要越界
    x_center, y_center = target_patch[1], target_patch[0]
    half_size = 128
    img = img[max(0, y_center-half_size):y_center+half_size, 
              max(0, x_center-half_size):x_center+half_size]
              
    # 裁剪后，你的 target 坐标在新的小图里需要重新计算，通常就是小图的中心
    target_patch = (half_size, half_size)




    # img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    # if img is None:
    #     raise FileNotFoundError(f"Cannot load image at {img_path}")
    
    # # 归一化并转为张量 [1, 1, H, W]
    # img_tensor = torch.from_numpy(img).float() / 255.0
    # img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    # OpenCV 读进来是 HWC (且为 BGR)，转为 RGB 并调整为 CHW 格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) # 形状变为 [1, 3, H, W]

    
    model.to(device)
    model.eval()

    # INFO: to reduce GPU memory consumption
    for param in model.parameters():
        param.requires_grad = False

    img_tensor = img_tensor.to(device).requires_grad_(True)
    
    baseline = torch.zeros_like(img_tensor).to(device)
    accumulated_grads = torch.zeros_like(img_tensor)
    
    # 路径积分 (Integrated Gradients)
    for i in range(steps):
        alpha = (i + 1) / steps
        interpolated = baseline + alpha * (img_tensor - baseline)
        interpolated = interpolated.detach().requires_grad_(True)
        
        out = model(interpolated)
        
        y, x = target_patch
        # 选取目标中心的一个小邻域来计算得分
        score = out[:, :, max(0, y-2):y+2, max(0, x-2):x+2].sum()
        
        model.zero_grad()
        score.backward()
        accumulated_grads += interpolated.grad

        # INFO: to reduce GPU memory consumption
        interpolated.grad = None 
        torch.cuda.empty_cache()
    
    lam = (img_tensor - baseline) * accumulated_grads
    lam = torch.abs(lam).detach().cpu().numpy().squeeze()
    
    if lam.ndim == 3:
        lam = np.sum(lam, axis=0)

    # 高斯平滑，让归因热力图更平滑
    lam = gaussian_filter(lam, sigma=2)
    lam = (lam - lam.min()) / (lam.max() - lam.min() + 1e-8)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.scatter([x], [y], c='red', s=40, marker='x') 
    plt.title(f'Input Image\nTarget: (x={x}, y={y})')
    
    plt.subplot(122)
    plt.imshow(lam, cmap='hot')
    plt.colorbar()
    plt.title('Local Attribution Map (LAM)')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"LAM image saved to {save_path}")
    else:
        plt.show()

# ---------------------------------------------------------
# 2. 命令行解析与执行
# ---------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize ERF and LAM for basicsr models")
    
    # 基础参数
    parser.add_argument('--model', type=str, required=True, choices=['wavelet_m3', 'ascnet', 'dls_nuc', 'idtransformer', 'mambairv2', 'restormer', 'mwdcnn', 'swinir', 'xformer', ''], default='wavelet_m3', help="Choose model type")
    parser.add_argument('--mode', type=str, required=True, choices=['erf', 'lam'], help="Choose visualization mode: 'erf' or 'lam'")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save the output plot (e.g., result.png)")
    parser.add_argument('--weight_path', type=str, required=True, help="Path to the .pth model weights")
    
    # ERF 专属参数
    parser.add_argument('--size', type=int, nargs=2, default=[256, 256], help="Input tensor size for ERF (H W). Default: 256 256")
    
    # LAM 专属参数
    parser.add_argument('--img_path', type=str, help="Path to the input noisy image (Required for LAM)")
    parser.add_argument('--target_x', type=int, default=128, help="Target X coordinate for LAM")
    parser.add_argument('--target_y', type=int, default=128, help="Target Y coordinate for LAM")
    
    args = parser.parse_args()

    # ==========================================
    # 🌟 修改区域：导入你的 basicsr 网络架构
    # ==========================================
    # 1. 导入你的网络类 (请根据实际路径修改)
    # 例如：from basicsr.archs.nafnet_arch import NAFNetLocal

    model_type = args.model

    print("Loading model architecture...")

    if model_type == 'wavelet_m3':
        from basicsr.archs.wavelet_m3_arch import wavelet_m3

        # 2. 实例化网络 (参数需与你训练时的 YAML 配置保持一致)
        # 因为是红外图像，确保 in_channels 和 out_channels 为 1

# network_g:
#   type: MambaIRv2
#   upscale: 1
#   in_chans: 3
#   img_size: 64
#   img_range: 1.
#   embed_dim: 64 # param 6.9M
#   d_state: 16
#   depths: [4, 4, 4,4,4,4]
#   num_heads: [4,4,4,4,4,4]
#   window_size: 16
#   inner_rank: 64
#   num_tokens: 32
#   convffn_kernel_size: 5
#   mlp_ratio: 2.

        model = wavelet_m3(
            upscale=1,
            in_chans=3,
            img_size=64,
            img_range=1.,
            embed_dim=64,
            d_state=16,
            depths=[4, 4, 4,4,4,4],
            num_heads=[4,4,4,4,4,4],
            window_size=16,
            inner_rank=64,
            num_tokens=32,
            convffn_kernel_size=5,
            mlp_ratio=2.,
        )

    elif model_type == 'ascnet':
        from basicsr.archs.ascnet_arch import ASCNet

# network_g:
#   type: ASCNet
#   in_ch: 3
#   out_ch: 3
#   feats: 64

        model = ASCNet(
            in_ch=3,
            out_ch=3,
            feats=64
        )

    elif model_type == 'dls_nuc':
        from basicsr.archs.DLS_NUC_arch import DLS_NUC

# network_g:
#   type: DLS_NUC
#   num_channels: 3
#   num_features: 256
#   scale: 1
#   kernel_size: 3

        model = DLS_NUC(
            num_channels=3,
            num_features=256,
            scale=1,
            kernel_size=3
        )

    elif model_type == 'idtransformer':
        from basicsr.archs.id_transformer_arch import IDTransformer

# network_g:
#   type: IDTransformer
#   inp_channels: 3
#   out_channels: 3
#   dim: 48
#   num_blocks: [4, 6, 6, 8]
#   num_refinement_blocks: 4
#   heads: [1, 2, 4, 8]
#   ffn_expansion_factor: 2.66
#   bias: false
#   LayerNorm_type: 'WithBias'

        model = IDTransformer(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias'
        )

    elif model_type == 'mambairv2':
        from basicsr.archs.mambairv2_arch import MambaIRv2

# network_g:
#   type: MambaIRv2
#   upscale: 1
#   in_chans: 3
#   img_size: 64
#   img_range: 1.
#   embed_dim: 32
#   d_state: 16
#   depths: [4, 4, 4,4,4,4]
#   num_heads: [4,4,4,4,4,4]
#   window_size: 16
#   inner_rank: 64
#   num_tokens: 128
#   convffn_kernel_size: 5
#   mlp_ratio: 2.

        model = MambaIRv2(
            upscale=1,
            in_chans=3,
            img_size=64,
            img_range=1.,
            embed_dim=32,
            d_state=16,
            depths=[4, 4, 4,4,4,4],
            num_heads=[4,4,4,4,4,4],
            window_size=16,
            inner_rank=64,
            num_tokens=128,
            convffn_kernel_size=5,
            mlp_ratio=2.,
        )

    elif model_type == 'restormer':
        from basicsr.archs.restormer_arch import Restormer

# network_g:
#   type: Restormer
#   inp_channels: 3
#   out_channels: 3
#   dim: 72
#   num_blocks: [4,6,6,8]
#   num_refinement_blocks: 4
#   heads: [1,2,4,8]
#   ffn_expansion_factor: 2.66
#   bias: False
#   LayerNorm_type: BiasFree
#   dual_pixel_task: False

        model = Restormer(
            inp_channels=3,
            out_channels=3,
            dim=72,
            num_blocks=[4,6,6,8],
            num_refinement_blocks=4,
            heads=[1,2,4,8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='BiasFree',
            dual_pixel_task=False
        )

    elif model_type == 'mwdcnn':
        from basicsr.archs.mwdcnn_arch import WMDCNN

# network_g:
#   type: WMDCNN
#   n_colors: 3
#   n_feats: 256
#   growth_rate: 32
#   rdb_num_layers: 16

        model = WMDCNN(
            n_colors=3,
            n_feats=256,
            growth_rate=32,
            rdb_num_layers=16
        )

    elif model_type == 'swinir':
        from basicsr.archs.swinir_arch import SwinIR


# network_g:
#   type: SwinIR
#   upscale: 1
#   in_chans: 3
#   img_size: 128
#   window_size: 8
#   img_range: 1.
#   depths: [6, 6, 6, 6, 6, 6]
#   embed_dim: 120
#   num_heads: [6, 6, 6, 6, 6, 6]
#   mlp_ratio: 2
#   upsampler: 'pixelshuffle'
#   resi_connection: '1conv'

        model = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=128,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=120,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )

    elif model_type == 'xformer':
        from basicsr.archs.x_former_arch import Xformer

# network_g:
#   type: Xformer
#   inp_channels: 3
#   out_channels: 3
#   dim: 48
#   num_blocks: [2, 4, 4]
#   spatial_num_blocks: [2,4,4,6]
#   num_refinement_blocks: 4
#   heads: [1, 2, 4, 8]
#   window_size: [16,16,16,16]
#   ffn_expansion_factor: 2.66
#   bias: False
#   LayerNorm_type: WithBias
#   dual_pixel_task: False

        model = Xformer(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[2, 4, 4],
            spatial_num_blocks=[2,4,4,6],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            window_size=[16,16,16,16],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            dual_pixel_task=False
        )

    # elif model_type == 'nafnet':
    #     from basicsr.archs.nafnet_arch import NAFNet
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 3. 加载预训练权重
    print(f"Loading weights from {args.weight_path}...")
    checkpoint = torch.load(args.weight_path, map_location=args.device)
    
    # 处理 basicsr 权重的两种常见存储格式
    if 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'], strict=True)
        print("Loaded EMA parameters.")
    elif 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=True)
        print("Loaded standard parameters.")
    else:
        model.load_state_dict(checkpoint, strict=True)
        print("Loaded bare parameters.")
    # ==========================================

    # 执行可视化
    if args.mode == 'erf':
        # input_shape = (1, 1, args.size[0], args.size[1])
        input_shape = (1, 3, args.size[0], args.size[1])
        # input_shape = (1, 3, 64, 64)
        print(f"Generating ERF with input size {input_shape}...")
        visualize_erf(model, input_size=input_shape, device=args.device, save_path=args.save_path)
        
    elif args.mode == 'lam':
        if not args.img_path:
            parser.error("--img_path is required when mode is 'lam'")
        print(f"Generating LAM for {args.img_path} at target (x={args.target_x}, y={args.target_y})...")
        visualize_lam(model, img_path=args.img_path, target_patch=(args.target_y, args.target_x), 
                      device=args.device, save_path=args.save_path)
