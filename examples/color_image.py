"""Example: TV denoising of a color image."""

import torch
from ftvp import proxtv


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, this library requires a GPU.")
        return

    C, H, W = 3, 256, 256
    img_clean = torch.zeros(C, H, W, device=device)
    img_clean[0, 64:192, 64:192] = 1.0
    img_clean[1, 96:160, 96:160] = 1.0
    img_clean[2, 32:224, 32:64] = 1.0

    noise_level = 0.15
    img_noisy = img_clean + noise_level * torch.randn_like(img_clean)

    lam = 0.08
    img_denoised = proxtv(img_noisy, lam)

    psnr_noisy = -10 * torch.log10(((img_noisy - img_clean) ** 2).mean()).item()
    psnr_denoised = -10 * torch.log10(((img_denoised - img_clean) ** 2).mean()).item()

    print(f"PSNR before: {psnr_noisy:.2f} dB")
    print(f"PSNR after:  {psnr_denoised:.2f} dB")


if __name__ == "__main__":
    main()
