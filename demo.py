#!/usr/bin/env python3
"""
PASS-SAM Demo: Single Image Unsupervised Semantic Segmentation

Usage:
    python demo.py --image path/to/image.jpg
    python demo.py --image path/to/image.jpg --output result.png --model r18

Requirements: pip install jittor pillow
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

import jittor as jt

# ── Configuration ──────────────────────────────────────────────
CHECKPOINTS = {
    "r18": "./weight/pass50_r18_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz256/checkpoint.pth.tar",
    "r34": "./weight/pass50_r34_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz384/checkpoint.pth.tar",
}
IMAGE_SIZE = {"r18": 256, "r34": 384}

COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128], [128, 64, 128], [0, 192, 128], [128, 192, 128],
    [64, 64, 0], [192, 64, 0], [64, 192, 0], [192, 192, 0],
    [64, 64, 128], [192, 64, 128], [64, 192, 128], [192, 192, 128],
    [0, 0, 64], [128, 0, 64], [0, 128, 64], [128, 128, 64],
    [0, 0, 192], [128, 0, 192], [0, 128, 192], [128, 128, 192],
    [64, 0, 64], [192, 0, 64], [64, 128, 64], [192, 128, 64],
    [64, 0, 192], [192, 0, 192], [64, 128, 192], [192, 128, 192],
    [0, 64, 64], [128, 64, 64], [0, 192, 64], [128, 192, 64],
], dtype=np.uint8)


def load_model(model_name="r18"):
    """Load PASS-SAM model from checkpoint."""
    assert model_name in CHECKPOINTS, f"Unknown model: {model_name}. Choose r18 or r34."
    checkpoint_path = CHECKPOINTS[model_name]
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print(f"   Download from: https://github.com/TYEclipse/PASS-SAM/releases")
        print(f"   Or run: wget <url> -O {checkpoint_path}")
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import src.resnet as resnet_models

    arch = f"resnet{model_name.replace('r','')}"
    model = resnet_models.__dict__[arch](
        normalize=True,
        hidden_mlp=2048,
        output_dim=256,
        nmb_prototypes=300,
        num_classes=50,
        patch_size=16,
        pretrained=False,
    )
    checkpoint = jt.load(checkpoint_path)
    model.load_parameters(checkpoint["model"])
    model.eval()
    return model


def segment_image(model, image_path, model_name="r18"):
    """Run unsupervised semantic segmentation on a single image."""
    from src.pseudo_sam_transforms import get_pseudo_sam_transform

    image = Image.open(image_path).convert("RGB")
    size = IMAGE_SIZE[model_name]
    
    transform = get_pseudo_sam_transform(size)
    img_tensor = transform(image).unsqueeze(0)
    
    with jt.no_grad():
        output = model(img_tensor)
    
    if isinstance(output, (list, tuple)):
        mask = output[0]
    else:
        mask = output
    
    mask = mask.squeeze().argmax(dim=0).numpy().astype(np.uint8)
    return mask, image.size


def colorize_mask(mask):
    """Apply colormap to mask for visualization."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(min(len(COLORMAP), mask.max() + 1)):
        color_mask[mask == label] = COLORMAP[label]
    return Image.fromarray(color_mask)


def main():
    parser = argparse.ArgumentParser(description="PASS-SAM Single Image Demo")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="output.png", help="Output image path")
    parser.add_argument("--model", default="r18", choices=["r18", "r34"],
                        help="Model variant (default: r18)")
    parser.add_argument("--overlay", action="store_true",
                        help="Overlay mask on original image")
    args = parser.parse_args()

    print(f"Loading PASS-SAM ({args.model})...")
    model = load_model(args.model)

    print(f"Segmenting: {args.image}")
    mask, original_size = segment_image(model, args.image, args.model)

    if args.overlay:
        original = Image.open(args.image).convert("RGB").resize(mask.shape[::-1])
        color = colorize_mask(mask)
        result = Image.blend(original, color, alpha=0.5)
    else:
        result = colorize_mask(mask)

    result.save(args.output)
    print(f"✅ Saved to: {args.output} ({result.size[0]}x{result.size[1]})")


if __name__ == "__main__":
    main()
