"""Example: TV denoising of an image."""

import torch
from ftvp import proxtv


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, this library requires a GPU.")
        return

    H, W = 256, 256
    img_clean = torch.zeros(H, W, device=device)
    img_clean[64:192, 64:192] = 1.0

    noise_level = 0.2
    img_noisy = img_clean + noise_level * torch.randn_like(img_clean)

    lam = 0.1
    img_denoised = proxtv(img_noisy, lam)

    mse_noisy = ((img_noisy - img_clean) ** 2).mean().item()
    mse_denoised = ((img_denoised - img_clean) ** 2).mean().item()

    print(f"MSE before: {mse_noisy:.4f}")
    print(f"MSE after:  {mse_denoised:.4f}")
    print(f"Improvement: {(1 - mse_denoised / mse_noisy) * 100:.1f}%")


if __name__ == "__main__":
    main()
