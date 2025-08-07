#!/usr/bin/env python3
"""
Panoramic Depth Generation Script
Generate depth maps for panoramic images using MoGe model
"""

import os
import torch
import numpy as np
import cv2
import argparse
from PIL import Image

# Set HuggingFace cache to use local models
os.environ['HF_HOME'] = '/mnt/pretrained'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/pretrained'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/pretrained'

# Import MoGe and HunyuanWorld depth utilities
from hy3dworld.utils.pano_depth_utils import build_depth_model, pred_pano_depth


def generate_panorama_depth(
    image_path: str, 
    output_path: str, 
    scale: float = 1.0,
    resize_to: int = 1920,
    verbose: bool = True,
    use_local_model: bool = True
):
    """
    Generate depth map for a panoramic image using MoGe model.
    
    Args:
        image_path (str): Path to the input panoramic image
        output_path (str): Directory to save the depth map
        scale (float): Scale factor for depth values
        resize_to (int): Target resolution for processing
        verbose (bool): Whether to print detailed information
    """
    
    # Check if input image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    if verbose:
        print(f"🖼️  Loading panoramic image from: {image_path}")
    
    # Load panoramic image
    pano_image = Image.open(image_path).convert("RGB")
    
    if verbose:
        print(f"📐 Image size: {pano_image.size}")
        print("🔧 Building MoGe depth model...")
    
    # Build MoGe depth model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_local_model and verbose:
        print(f"📁 Using local cache directory: /mnt/pretrained")
    
    # Build model (will use local cache if available)
    depth_model = build_depth_model(device)
    
    if verbose:
        print(f"🚀 Using device: {device}")
        print("📊 Generating panoramic depth map...")
    
    # Generate depth map
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    depth_result = pred_pano_depth(
        model=depth_model,
        image=pano_image,
        img_name=img_name,
        scale=scale,
        resize_to=resize_to,
        remove_pano_depth_nan=True,
        last_layer_mask=None,
        last_layer_depth=None,
        verbose=verbose
    )
    
    # Extract depth information
    depth_map = depth_result["distance"].cpu().numpy()
    depth_mask = depth_result["mask"]
    rgb_image = depth_result["rgb"].cpu().numpy()
    
    if verbose:
        print(f"✅ Depth map generated with shape: {depth_map.shape}")
        print(f"📈 Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
    
    # Save results
    depth_output_path = os.path.join(output_path, f"{img_name}_depth.npy")
    depth_vis_path = os.path.join(output_path, f"{img_name}_depth_vis.png")
    mask_path = os.path.join(output_path, f"{img_name}_mask.png")
    
    # Save depth as numpy array
    np.save(depth_output_path, depth_map)
    
    # Create depth visualization (normalized to 0-255)
    depth_vis = depth_map.copy()
    depth_vis[~depth_mask] = np.nan  # Set invalid areas to NaN
    
    # Normalize depth for visualization
    valid_depth = depth_vis[depth_mask]
    if len(valid_depth) > 0:
        depth_min, depth_max = np.percentile(valid_depth, [5, 95])  # Robust normalization
        depth_vis_norm = np.clip((depth_vis - depth_min) / (depth_max - depth_min), 0, 1)
        depth_vis_norm[~depth_mask] = 0  # Set invalid areas to black
        
        # Convert to 8-bit and apply colormap
        depth_vis_8bit = (depth_vis_norm * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_vis_8bit, cv2.COLORMAP_INFERNO)
        cv2.imwrite(depth_vis_path, depth_colormap)
    
    # Save mask
    mask_vis = (depth_mask * 255).astype(np.uint8)
    cv2.imwrite(mask_path, mask_vis)
    
    if verbose:
        print("💾 Saved files:")
        print(f"   - Depth array: {depth_output_path}")
        print(f"   - Depth visualization: {depth_vis_path}")
        print(f"   - Valid mask: {mask_path}")
    
    return depth_result


def main():
    parser = argparse.ArgumentParser(description="Generate depth map for panoramic images using MoGe")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input panoramic image")
    parser.add_argument("--output_path", type=str, default="depth_output",
                        help="Directory to save the depth map outputs")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for depth values")
    parser.add_argument("--resize_to", type=int, default=1920,
                        help="Target resolution for processing (width)")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed processing information")
    parser.add_argument("--use_local", action="store_true", default=True,
                        help="Use local MoGe model instead of downloading")
    
    args = parser.parse_args()
    
    print("🌟 Panoramic Depth Generation with MoGe")
    print("=" * 50)
    
    try:
        result = generate_panorama_depth(
            image_path=args.image_path,
            output_path=args.output_path,
            scale=args.scale,
            resize_to=args.resize_to,
            verbose=args.verbose,
            use_local_model=args.use_local
        )
        
        print("🎉 Depth generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during depth generation: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())