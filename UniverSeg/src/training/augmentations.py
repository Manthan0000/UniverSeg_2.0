import math
import random
from typing import Tuple

import torch
import torch.nn.functional as F


def _gaussian_kernel_2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    return kernel.view(1, 1, kernel_size, kernel_size)


def apply_gaussian_blur(img: torch.Tensor, sigma: float, kernel_size: int = 5) -> torch.Tensor:
    """Applies 2D Gaussian blur to a tensor of shape (..., 1, H, W)."""
    orig_shape = img.shape
    flat = img.reshape(-1, 1, orig_shape[-2], orig_shape[-1])
    pad = kernel_size // 2
    kernel = _gaussian_kernel_2d(kernel_size, sigma, flat.device, flat.dtype)
    padded = F.pad(flat, (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(padded, kernel)
    return blurred.reshape(orig_shape)


def apply_shared_affine(
    images: torch.Tensor,
    masks: torch.Tensor,
    angle_deg: float,
    tx: float,
    ty: float,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the exact same 2D affine transform to a stack of images (N, 1, H, W)
    and masks (N, 1, H, W). Uses bilinear interpolation for images and nearest for masks.
    """
    N, C, H, W = images.shape
    theta_rad = math.radians(angle_deg)
    cos_t = math.cos(theta_rad) / scale
    sin_t = math.sin(theta_rad) / scale

    affine_mat = torch.tensor(
        [
            [cos_t, -sin_t, tx],
            [sin_t, cos_t, ty],
        ],
        dtype=images.dtype,
        device=images.device,
    ).unsqueeze(0).repeat(N, 1, 1)

    grid = F.affine_grid(affine_mat, size=(N, C, H, W), align_corners=False)
    warped_images = F.grid_sample(
        images, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    warped_masks = F.grid_sample(
        masks.float(), grid, mode="nearest", padding_mode="border", align_corners=False
    )
    return warped_images, (warped_masks > 0.5).float()


def apply_recipe_r_augmentations(
    query_image: torch.Tensor,
    query_label: torch.Tensor,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies Recipe R (light) task-level and in-task augmentations.
    Shapes:
      query_image:    (1, H, W)
      query_label:    (1, H, W)
      support_images: (S, 1, H, W)
      support_labels: (S, 1, H, W)
    Returns augmented tensors of the exact same shapes with images in [0, 1]
    and masks strictly binary {0.0, 1.0}.
    """
    S = support_images.shape[0]

    # Stack query + supports into (S + 1, 1, H, W) for shared task-level ops
    all_imgs = torch.cat([query_image.unsqueeze(0), support_images], dim=0).clone()
    all_msks = torch.cat([query_label.unsqueeze(0), support_labels], dim=0).clone().float()

    # 1. Task-level horizontal flip (p=0.5)
    if rng.random() < 0.5:
        all_imgs = torch.flip(all_imgs, dims=[-1])
        all_msks = torch.flip(all_msks, dims=[-1])

    # 2. Task-level vertical flip (p=0.5)
    if rng.random() < 0.5:
        all_imgs = torch.flip(all_imgs, dims=[-2])
        all_msks = torch.flip(all_msks, dims=[-2])

    # 3. Task-level affine transform (p=0.5)
    if rng.random() < 0.5:
        angle = rng.uniform(0.0, 360.0)
        tx = rng.uniform(-0.2, 0.2)
        ty = rng.uniform(-0.2, 0.2)
        scale = rng.uniform(0.8, 1.1)
        all_imgs, all_msks = apply_shared_affine(all_imgs, all_msks, angle, tx, ty, scale)

    # 4. Task-level intensity inversion 1 - x on images only (p=0.5)
    if rng.random() < 0.5:
        all_imgs = 1.0 - all_imgs

    # 5. Task-level label flip 1 - m on masks only (p=0.5)
    if rng.random() < 0.5:
        all_msks = 1.0 - all_msks

    # 6. In-task independent per-example augmentations on images
    for i in range(S + 1):
        img_i = all_imgs[i]

        # Brightness / contrast (p=0.25)
        if rng.random() < 0.25:
            brightness = rng.uniform(-0.1, 0.1)
            contrast = rng.uniform(0.5, 1.5)
            mean_val = img_i.mean()
            img_i = (img_i - mean_val) * contrast + mean_val + brightness

        # Gaussian noise (p=0.25)
        if rng.random() < 0.25:
            sigma = rng.uniform(0.005, 0.05)
            gen = torch.Generator(device=img_i.device)
            gen.manual_seed(rng.randint(0, 2**31 - 1))
            noise = torch.randn(img_i.shape, generator=gen, device=img_i.device, dtype=img_i.dtype) * sigma
            img_i = img_i + noise

        # Gaussian blur (p=0.25)
        if rng.random() < 0.25:
            sigma = rng.uniform(0.1, 1.1)
            img_i = apply_gaussian_blur(img_i, sigma=sigma, kernel_size=5)

        all_imgs[i] = img_i

    # Clamp images to [0, 1] and enforce strictly binary masks {0, 1}
    all_imgs = all_imgs.clamp(0.0, 1.0)
    all_msks = (all_msks > 0.5).float()

    q_img_out = all_imgs[0]
    q_msk_out = all_msks[0]
    s_imgs_out = all_imgs[1:]
    s_msks_out = all_msks[1:]

    return q_img_out, q_msk_out, s_imgs_out, s_msks_out
