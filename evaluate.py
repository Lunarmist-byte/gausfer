"""
Gausfer Evaluation Script
Computes PSNR, SSIM, and LPIPS metrics by rendering all training views
and comparing against ground truth images.

Usage:
    venv\Scripts\python evaluate.py
    venv\Scripts\python evaluate.py --checkpoint ./output/gauss_checkpoint.pth
"""
import os
import sys
import math
import torch
import numpy as np
from PIL import Image

from dataset_loader import RoomDatasetLoader
from gaussian_model import GaussianModel
from rasterizer import RoomRasterizerCUDA
from nerf_model import RoomNeRF
from main import ViewCamera

def compute_psnr(img1, img2):
    """Peak Signal-to-Noise Ratio between two images (torch tensors, range [0,1])"""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(1.0 / mse)

def compute_ssim_metric(img1, img2, window_size=11):
    """Structural Similarity Index (expects [H, W, 3] tensors)"""
    from loss_utils import ssim
    # ssim expects [B, C, H, W]
    i1 = img1.permute(2, 0, 1).unsqueeze(0)
    i2 = img2.permute(2, 0, 1).unsqueeze(0)
    return ssim(i1, i2).item()

def compute_lpips_metric(img1, img2, lpips_fn):
    """Learned Perceptual Image Patch Similarity (lower = better)"""
    # lpips expects [B, C, H, W] in range [0,1] or [-1,1] depending on version
    i1 = img1.permute(2, 0, 1).unsqueeze(0)
    i2 = img2.permute(2, 0, 1).unsqueeze(0)
    # Normalize to [-1, 1] as LPIPS expects
    i1 = i1 * 2.0 - 1.0
    i2 = i2 * 2.0 - 1.0
    with torch.no_grad():
        return lpips_fn(i1, i2).item()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Gausfer reconstruction quality")
    parser.add_argument("--checkpoint", type=str, default="./output/gauss_checkpoint.pth",
                        help="Path to Gaussian checkpoint")
    parser.add_argument("--images_dir", type=str, default="./images",
                        help="Path to images directory")
    parser.add_argument("--colmap_dir", type=str, default="./output/sparse/0",
                        help="Path to COLMAP model directory")
    parser.add_argument("--output", type=str, default="./output/eval_results.txt",
                        help="Path to save evaluation results")
    parser.add_argument("--max_views", type=int, default=0,
                        help="Max views to evaluate (0 = all)")
    parser.add_argument("--save_renders", action="store_true",
                        help="Save rendered images to output/renders/")
    args = parser.parse_args()

    print("=" * 60)
    print("  GAUSFER EVALUATION")
    print("=" * 60)

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Run training first, then evaluate.")
        sys.exit(1)

    # Load dataset
    print(f"\nLoading dataset from {args.colmap_dir}...")
    dataset = RoomDatasetLoader(colmap_dir=args.colmap_dir, images_dir=args.images_dir)
    num_views = len(dataset.images)
    if args.max_views > 0:
        num_views = min(num_views, args.max_views)
    print(f"  Evaluating {num_views} views")

    # Load model
    print(f"\nLoading Gaussian model from {args.checkpoint}...")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_checkpoint(args.checkpoint)
    # Set max SH degree since we're evaluating a fully trained model
    gaussians.active_sh_degree = gaussians.max_sh_degree
    print(f"  Loaded {gaussians.xyz.shape[0]} Gaussians")
    print(f"  Active SH degree: {gaussians.active_sh_degree}")

    # Setup rasterizer
    rasterizer = RoomRasterizerCUDA()

    # Try loading LPIPS
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').cuda()
        print("  LPIPS (VGG) loaded successfully")
    except ImportError:
        print("  WARNING: lpips not installed. Skipping LPIPS metric.")
        print("  Install with: pip install lpips")

    # Create renders directory if requested
    if args.save_renders:
        renders_dir = os.path.join("output", "renders")
        os.makedirs(renders_dir, exist_ok=True)
        print(f"  Saving renders to {renders_dir}/")

    # Evaluate
    print(f"\nRendering and evaluating {num_views} views...\n")
    psnr_list = []
    ssim_list = []
    lpips_list = []

    for i in range(num_views):
        gt_image, (H, W, K, c2w) = dataset.get_training_batch(i)
        view_cam = ViewCamera(H, W, K, c2w)

        # Render
        with torch.no_grad():
            render_pkg = rasterizer.render_room_view(view_cam, gaussians)
            rendered = render_pkg["render"].permute(1, 2, 0).clamp(0, 1)  # [H, W, 3]

        # Compute metrics
        psnr_val = compute_psnr(rendered, gt_image)
        ssim_val = compute_ssim_metric(rendered, gt_image)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        lpips_val = None
        if lpips_fn is not None:
            lpips_val = compute_lpips_metric(rendered, gt_image, lpips_fn)
            lpips_list.append(lpips_val)

        # Progress
        lpips_str = f" | LPIPS: {lpips_val:.4f}" if lpips_val is not None else ""
        print(f"  View {i+1:3d}/{num_views} | PSNR: {psnr_val:6.2f} dB | SSIM: {ssim_val:.4f}{lpips_str}")

        # Save rendered image
        if args.save_renders:
            img_np = (rendered.cpu().numpy() * 255).astype(np.uint8)
            img_name = dataset.images[i]['name']
            Image.fromarray(img_np).save(os.path.join(renders_dir, f"render_{img_name}"))

    # Aggregate results
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Views evaluated:  {num_views}")
    print(f"  Gaussian count:   {gaussians.xyz.shape[0]}")
    print(f"  SH degree:        {gaussians.active_sh_degree}")
    print(f"")
    print(f"  Mean PSNR:   {avg_psnr:.2f} dB")
    print(f"  Mean SSIM:   {avg_ssim:.4f}")

    if lpips_list:
        avg_lpips = np.mean(lpips_list)
        print(f"  Mean LPIPS:  {avg_lpips:.4f}")
    else:
        avg_lpips = None

    print(f"")
    print(f"  Best PSNR:   {max(psnr_list):.2f} dB (View {np.argmax(psnr_list)+1})")
    print(f"  Worst PSNR:  {min(psnr_list):.2f} dB (View {np.argmin(psnr_list)+1})")
    print("=" * 60)

    # Save results to file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write("GAUSFER EVALUATION RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Views: {num_views}\n")
        f.write(f"Gaussians: {gaussians.xyz.shape[0]}\n")
        f.write(f"SH degree: {gaussians.active_sh_degree}\n\n")
        f.write(f"Mean PSNR:  {avg_psnr:.2f} dB\n")
        f.write(f"Mean SSIM:  {avg_ssim:.4f}\n")
        if avg_lpips is not None:
            f.write(f"Mean LPIPS: {avg_lpips:.4f}\n")
        f.write(f"\nPer-view results:\n")
        f.write(f"{'View':>6} | {'PSNR (dB)':>10} | {'SSIM':>8}")
        if lpips_list:
            f.write(f" | {'LPIPS':>8}")
        f.write("\n")
        f.write("-" * 45 + "\n")
        for i in range(num_views):
            f.write(f"{i+1:6d} | {psnr_list[i]:10.2f} | {ssim_list[i]:8.4f}")
            if lpips_list:
                f.write(f" | {lpips_list[i]:8.4f}")
            f.write("\n")

    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
