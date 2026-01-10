"""Example: TV denoising of a real image from file."""

import argparse
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from ftvp import proxtv


def load_image(path: str, device: torch.device) -> torch.Tensor:
    """Load image as (C, H, W) tensor normalized to [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    return tensor


def save_image(tensor: torch.Tensor, path: str):
    """Save (C, H, W) tensor to file."""
    arr = tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def pad_to_even(img: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad image to even dimensions, return padded image and original size."""
    C, H, W = img.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    return img, (H, W)


def unpad(img: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
    """Remove padding to restore original size."""
    H, W = original_size
    return img[:, :H, :W]


def main():
    parser = argparse.ArgumentParser(description="TV denoising of an image")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path (default: input_denoised.png)")
    parser.add_argument("-l", "--lambda", dest="lam", type=float, default=0.1,
                        help="Regularization parameter (default: 0.1)")
    parser.add_argument("--add-noise", type=float, default=0.0,
                        help="Add Gaussian noise with this std before denoising")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not available, this will be slow or fail.")
        return

    print(f"Loading {args.input}...")
    img = load_image(args.input, device)
    print(f"Image size: {img.shape[1]}x{img.shape[2]} ({img.shape[0]} channels)")

    if args.add_noise > 0:
        print(f"Adding Gaussian noise (std={args.add_noise})...")
        img = img + args.add_noise * torch.randn_like(img)

    img_padded, orig_size = pad_to_even(img)

    print(f"Denoising with lambda={args.lam}...")
    denoised = proxtv(img_padded, args.lam)

    denoised = unpad(denoised, orig_size)

    output_path = args.output
    if output_path is None:
        p = Path(args.input)
        output_path = str(p.parent / f"{p.stem}_denoised{p.suffix}")

    print(f"Saving to {output_path}...")
    save_image(denoised, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
