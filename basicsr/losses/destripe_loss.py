import torch
import torch.nn.functional as F
from torch import nn as nn
from pytorch_wavelets import DWTForward, DWTInverse

from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class HBGMLoss(nn.Module):
    """Haar Wavelet Background Guidance Module (HBGM) Loss."""
    def __init__(self, loss_weight=1.0):
        super(HBGMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.dwt = DWTForward(J=3, wave='haar')
        self.idwt = DWTInverse(wave='haar')

    def forward(self, a, b):
        self.dwt = self.dwt.to(a.device)
        self.idwt = self.idwt.to(a.device)
        
        yl_a, yh_a = self.dwt(a)
        yl_a.zero_()
        for i in range(len(yh_a)):
            yh_a[i][:, :, 1, :, :].zero_()
        out_a = self.idwt((yl_a, yh_a))

        yl_b, yh_b = self.dwt(b)
        yl_b.zero_()
        for i in range(len(yh_b)):
            yh_b[i][:, :, 1, :, :].zero_()
        out_b = self.idwt((yl_b, yh_b))

        # Mean Square Error (matches usage in original code)
        loss = F.mse_loss(out_a, out_b)
        return self.loss_weight * loss

@LOSS_REGISTRY.register()
class DreclossStripe(nn.Module):
    """Stripe preservation loss."""
    def __init__(self, loss_weight=1.0):
        super(DreclossStripe, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        h_x = x.size()[2]
        h_y = y.size()[2]
        h_tv_x = (x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
        h_tv_y = (y[:, :, 1:, :] - y[:, :, :h_y - 1, :])
        loss = F.l1_loss(h_tv_x, h_tv_y)
        return self.loss_weight * loss

@LOSS_REGISTRY.register()
class MSSSIML1Loss(nn.Module):
    """Mixed MS-SSIM and L1 Loss."""
    def __init__(self, loss_weight=1.0, alpha=0.025, channel=3):
        super(MSSSIML1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.channel = channel
        # Simplified version or port the full one from utils.py if needed.
        # For now, implementing the logic from MS_SSIM_L1_LOSS in utils.py.
        self.gaussian_sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]
        filter_size = int(4 * self.gaussian_sigmas[-1] + 1)
        self.register_buffer('g_masks', torch.zeros((self.channel * len(self.gaussian_sigmas), 1, filter_size, filter_size)))
        for idx, sigma in enumerate(self.gaussian_sigmas):
            mask = self._fspecial_gauss_2d(filter_size, sigma)
            for c in range(self.channel):
                self.g_masks[self.channel * idx + c, 0, :, :] = mask

    def _fspecial_gauss_1d(self, size, sigma):
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g

    def _fspecial_gauss_2d(self, size, sigma):
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        pad = int(2 * self.gaussian_sigmas[-1])
        
        mux = F.conv2d(x, self.g_masks, groups=c, padding=pad)
        muy = F.conv2d(y, self.g_masks, groups=c, padding=pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=c, padding=pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=c, padding=pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=c, padding=pad) - muxy

        C1 = 0.01**2
        C2 = 0.03**2
        l = (2 * muxy + C1) / (mux2 + muy2 + C1)
        cs = (2 * sigmaxy + C2) / (sigmax2 + sigmay2 + C2)
        
        if self.channel == 3:
            lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
            PIcs = cs.prod(dim=1)
        else:
            lM = l[:, -1, :, :]
            PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs
        loss_l1 = F.l1_loss(x, y, reduction='none')
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.channel, length=self.channel),
                               groups=c, padding=pad).mean(1)

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1
        return self.loss_weight * loss_mix.mean()
