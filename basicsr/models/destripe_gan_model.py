import torch
import numpy as np
from collections import OrderedDict
from torch import nn as nn

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

class SGM(nn.Module):
    """Stripe Simulation Model (SGM) Case 3."""
    def __init__(self, noise_range=[0.02, 0.12]):
        super(SGM, self).__init__()
        self.noise_range = noise_range

    def forward(self, img):
        # img is [B, C, H, W] in [0, 1]
        noise_S = torch.zeros_like(img)
        b, c, h, w = img.size()
        
        beta1 = np.random.uniform(self.noise_range[0], self.noise_range[1], size=b)
        beta2 = np.random.uniform(self.noise_range[0], self.noise_range[1], size=b)
        beta3 = np.random.uniform(self.noise_range[0], self.noise_range[1], size=b)
        beta4 = np.random.uniform(self.noise_range[0], self.noise_range[1], size=b)

        for m in range(b):
            A1 = np.random.normal(0, beta1[m], size=w)
            A2 = np.random.normal(0, beta2[m], size=w)
            A3 = np.random.normal(0, beta3[m], size=w)
            A4 = np.random.normal(0, beta4[m], size=w)
            
            # Tile to match image size
            A1 = np.tile(A1, (h, 1))
            A2 = np.tile(A2, (h, 1))
            A3 = np.tile(A3, (h, 1))
            A4 = np.tile(A4, (h, 1))
            
            A1 = torch.from_numpy(A1).to(img.device).float()
            A2 = torch.from_numpy(A2).to(img.device).float()
            A3 = torch.from_numpy(A3).to(img.device).float()
            A4 = torch.from_numpy(A4).to(img.device).float()
            
            # SGM Case 3 Logic
            img_m = img[m]
            imgn_m = A1 + A2 * img_m + A3 * A3 * img_m + A4 * A4 * A4 * img_m + img_m
            noise_S[m] = torch.clamp(imgn_m, 0., 1.)
            
        return noise_S

@MODEL_REGISTRY.register()
class DestripeGANModel(SRModel):
    """DestripeCycleGAN Model."""

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        self.sgm = SGM(noise_range=train_opt.get('sgm_noise_range', [0.02, 0.12]))

        self.net_g.train()
        self.net_d.train()

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('hbgm_opt'):
            self.cri_hbgm = build_loss(train_opt['hbgm_opt']).to(self.device)
        else:
            self.cri_hbgm = None

        if train_opt.get('stripe_opt'):
            self.cri_stripe = build_loss(train_opt['stripe_opt']).to(self.device)
        else:
            self.cri_stripe = None

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        
        real_A_norm = self.lq * 2.0 - 1.0
        self.output = self.net_g(real_A_norm)
        fake_B_unnorm = (self.output + 1.0) / 2.0
        
        self.recon_A = self.sgm(fake_B_unnorm)
        
        self.fake_A = self.sgm(self.gt)
        fake_A_norm = self.fake_A * 2.0 - 1.0
        self.real_B_recon = self.net_g(fake_A_norm)
        
        real_B_norm = self.gt * 2.0 - 1.0
        self.fake_B_I = self.net_g(real_B_norm)

        l_g_total = 0
        loss_dict = OrderedDict()
        
        # Use first scale of MultiScaleDis as in original code backward_EG
        fake_B_pred = self.net_d(self.output)[0]
        real_B_recon_pred = self.net_d(self.real_B_recon)[0]
        l_g_gan = (self.cri_gan(fake_B_pred, True, is_disc=False) + 
                   self.cri_gan(real_B_recon_pred, True, is_disc=False)) * 0.5
        l_g_total += l_g_gan
        loss_dict['l_g_gan'] = l_g_gan

        if self.cri_hbgm:
            l_g_hbgm = self.cri_hbgm(real_A_norm, self.output)
            l_g_total += l_g_hbgm
            loss_dict['l_g_hbgm'] = l_g_hbgm

        if self.cri_stripe:
            # recon_A should match real_A (lq). original code uses TVloss(real_A_recon_norm, real_A_train)
            recon_A_norm = self.recon_A * 2.0 - 1.0
            l_g_cycle_A = self.cri_stripe(recon_A_norm, real_A_norm)
            l_g_total += l_g_cycle_A
            loss_dict['l_g_cycle_A'] = l_g_cycle_A
            
        if self.cri_pix:
            gt_norm = self.gt * 2.0 - 1.0
            l_g_cycle_B = self.cri_pix(self.real_B_recon, gt_norm)
            l_g_total += l_g_cycle_B
            loss_dict['l_g_cycle_B'] = l_g_cycle_B

            l_g_id = self.cri_pix(self.fake_B_I, gt_norm)
            l_g_total += l_g_id
            loss_dict['l_g_id'] = l_g_id

        if self.cri_perceptual:
            l_g_percep, _ = self.cri_perceptual(self.output, real_A_norm)
            l_g_total += l_g_percep
            loss_dict['l_g_percep'] = l_g_percep

        l_g_total.backward()
        self.optimizer_g.step()

        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        
        gt_norm = self.gt * 2.0 - 1.0
        real_d_pred = self.net_d(gt_norm)
        l_d_real = 0
        for pred in real_d_pred:
            l_d_real += self.cri_gan(pred, True, is_disc=True)
        l_d_real /= len(real_d_pred)
        loss_dict['l_d_real'] = l_d_real
        l_d_real.backward()

        # fake_B and real_B_recon are fake. Original code used pool of 50.
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake1 = 0
        for pred in fake_d_pred:
            l_d_fake1 += self.cri_gan(pred, False, is_disc=True)
        l_d_fake1 /= len(fake_d_pred)
        
        recon_B_pred = self.net_d(self.real_B_recon.detach())
        l_d_fake2 = 0
        for pred in recon_B_pred:
            l_d_fake2 += self.cri_gan(pred, False, is_disc=True)
        l_d_fake2 /= len(recon_B_pred)
        
        l_d_fake = (l_d_fake1 + l_d_fake2) * 0.5
        loss_dict['l_d_fake'] = l_d_fake
        l_d_fake.backward()
        
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['result'] = (self.output.detach().cpu() + 1.0) / 2.0
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
