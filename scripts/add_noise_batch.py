import os
import argparse
import cv2
import numpy as np
import torch
import random
from tqdm import tqdm


def add_full_mixed_noise(
    img_tensor,
    stripe_intensity=[0.004, 0.02],
    shot_intensity=0.002,
    low_freq_intensity=0.1,
    smoothness=2,
    low_freq_scale=120
):
    """
    Adds mixed noise to an image, including smooth stripes, signal-dependent shot noise,
    and low-frequency noise.

    Args:
        img_tensor (torch.Tensor): Input tensor of shape [C, H, W] with values in [0, 1].
        stripe_intensity (list): The intensity range for stripe noise.
        shot_intensity (float): The intensity factor for shot noise.
        low_freq_intensity (float): The intensity factor for low-frequency noise.
        smoothness (int): The smoothness factor for stripe noise (higher value means smoother).
        low_freq_scale (int): The scale factor for low-frequency noise (higher value means
                              slower variation / larger patches).
    Returns:
        torch.Tensor: The noisy image tensor of shape [C, H, W] with values in [0, 1].
    """
    C, H, W = img_tensor.shape
    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]

    # --- 1. Generate smooth stripe noise ---
    def generate_smooth_template(beta, width, height, smooth_factor):
        low_res_w = max(2, width // smooth_factor)
        low_res_noise = np.random.normal(0, beta, size=(1, low_res_w)).astype(np.float32)
        smooth_noise_row = cv2.resize(low_res_noise, (width, 1), interpolation=cv2.INTER_CUBIC)
        return torch.from_numpy(np.tile(smooth_noise_row, (height, 1))).float()

    beta1 = np.random.uniform(stripe_intensity[0], stripe_intensity[1])
    stripe_template = generate_smooth_template(beta1, W, H, smoothness)

    # --- 2. Generate base map for signal-dependent shot noise ---
    shot_noise_map = torch.randn_like(img_tensor[0, 0]) # Shape: [H, W]

    # --- 3. Generate low-frequency noise surface ---
    low_res_h = max(2, H // low_freq_scale)
    low_res_w = max(2, W // low_freq_scale)

    # Generate a low-resolution random "height map"
    # Using uniform is more like a slowly varying bias field
    low_res_map = np.random.uniform(-1, 1, size=(low_res_h, low_res_w)).astype(np.float32)

    # Smoothly upscale to full size using bicubic interpolation
    smooth_low_freq_map = cv2.resize(low_res_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # Multiply by intensity factor and convert to Tensor
    low_freq_noise = torch.from_numpy(smooth_low_freq_map) * low_freq_intensity

    # --- 4. Combine all noise types ---
    noisy_img_tensor = img_tensor.clone()
    for c in range(C):
        img = img_tensor[0, c]

        # Calculate stripe noise component (signal-dependent)
        stripe_noise = stripe_template * img

        # Calculate shot noise component (signal-dependent)
        shot_noise_std = torch.sqrt(torch.clamp(img, min=0) * shot_intensity)
        shot_noise = shot_noise_map * shot_noise_std

        # Add all components to the original image
        # Low-frequency noise is additive and superimposed directly onto the image
        noisy = img + stripe_noise + shot_noise + low_freq_noise

        noisy_img_tensor[0, c] = torch.clip(noisy, 0., 1.)

    return noisy_img_tensor.squeeze(0)  # [C, H, W]


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
    return img_tensor


def save_image(tensor, path):
    img = tensor.permute(1, 2, 0).numpy()  # [H, W, C]
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    img_names = [f for f in os.listdir(args.in_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for name in tqdm(img_names, desc="Adding noise"):
        img_path = os.path.join(args.in_dir, name)
        out_path = os.path.join(args.out_dir, name)

        img_tensor = load_image(img_path)
        noisy_tensor = add_full_mixed_noise(
            img_tensor,
            stripe_intensity=[0.004, 0.02],
            shot_intensity=0.002,
            low_freq_intensity=0.10,
            smoothness=2,
            low_freq_scale=120
        )

        save_image(noisy_tensor, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate noisy images using custom noise model.")
    parser.add_argument('--in_dir', type=str, required=True, help='Directory of clean input images')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save noisy images')
    parser.add_argument('--noise_range', nargs=2, type=float, default=[0.05, 0.15], help='Noise intensity range')

    args = parser.parse_args()
    main(args)
