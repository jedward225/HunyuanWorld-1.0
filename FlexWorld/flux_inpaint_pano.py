#!/usr/bin/env python3
"""
FLUX Panorama Inpainting - based on flux_inpaint_simple.py
Specialized for panoramic image completion
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
import sys
import os

# Add FLUX path
sys.path.append('/home/liujiajun/HunyuanWorld-1.0/FLUX-Controlnet-Inpainting')

from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

check_min_version("0.30.2")

def flux_inpaint_panorama(image_path, mask_path, prompt="panoramic street scene", output_path="flux_inpainted_pano.png"):
    """
    Inpaint panoramic image using FLUX Controlnet
    
    Args:
        image_path: Path to panoramic image
        mask_path: Path to inpainting mask (black=inpaint, white=keep)
        prompt: Text prompt for inpainting
        output_path: Output path for result
    """
    
    print("🎨 Loading FLUX models for panorama inpainting...")
    
    # Build pipeline exactly like flux_inpaint_simple.py
    controlnet = FluxControlNetModel.from_pretrained(
        "/mnt2/FLUX.1-dev-Controlnet-Inpainting-Alpha", 
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    
    transformer = FluxTransformer2DModel.from_pretrained(
        "/mnt/pretrained/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21", 
        subfolder='transformer', 
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "/mnt/pretrained/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to("cuda")
    
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    
    print("✅ Models loaded successfully!")
    
    # Load panoramic image and mask
    print(f"📷 Loading panorama: {image_path}")
    print(f"🎭 Loading mask: {mask_path}")
    
    # Load images - keep original size for panorama processing
    original_image = load_image(image_path).convert("RGB")
    original_size = original_image.size
    print(f"  Original panorama size: {original_size}")
    
    # Load mask
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Panorama-specific processing
    # For panoramic images, we might need different sizing strategy
    # Common panorama ratios: 2:1 (360°×180°) or specific aspect ratios
    
    # Option 1: Resize to FLUX-friendly size while preserving aspect ratio
    target_height = 768
    aspect_ratio = original_size[0] / original_size[1]
    target_width = int(target_height * aspect_ratio)
    
    # Ensure width is multiple of 8 (FLUX requirement)
    target_width = (target_width // 8) * 8
    target_size = (target_width, target_height)
    
    print(f"  Target processing size: {target_size}")
    
    # Resize image and mask
    image = original_image.resize(target_size, Image.LANCZOS)
    
    # Process mask: mask_for_inpainting.png是透明度mask (白色=有coverage, 黑色=无coverage)
    # 使用threshold方式，就像flux_inpaint_simple.py一样
    flux_mask = np.zeros_like(mask_image)
    threshold = 6  # 使用与flux_inpaint_simple.py相同的threshold
    flux_mask[mask_image <= threshold] = 255  # 低coverage区域设为白色（需要inpaint）
    
    # Apply dilation to expand mask coverage
    kernel = np.ones((3,3), np.uint8)
    flux_mask = cv2.dilate(flux_mask, kernel, iterations=2)
    
    print(f"  Panorama mask analysis:")
    print(f"    Need inpaint (≤{threshold}): {np.sum(mask_image <= threshold)} pixels")
    print(f"    After dilation: {np.sum(flux_mask > 0)} pixels") 
    print(f"    Has geometry (>{threshold}): {np.sum(mask_image > threshold)} pixels")
    print(f"    Final inpaint ratio: {np.sum(flux_mask > 0)/np.prod(mask_image.shape)*100:.2f}%")
    
    # Convert mask to PIL and resize
    mask = Image.fromarray(flux_mask).convert("RGB").resize(target_size, Image.NEAREST)
    
    # Set random seed for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(42)
    
    print(f"🎨 Starting panorama inpainting with prompt: '{prompt}'")
    print(f"    Processing size: {target_size}")
    
    # Inpaint with panorama-optimized parameters
    result = pipe(
        prompt=prompt,
        height=target_size[1],
        width=target_size[0],
        control_image=image,
        control_mask=mask,
        num_inference_steps=25,  # Slightly more steps for better quality
        generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="blurry, distorted, low quality, artifacts",
        true_guidance_scale=1.0
    ).images[0]
    
    # Resize result back to original panorama dimensions
    result_original_size = result.resize(original_size, Image.LANCZOS)
    
    return result_original_size, original_image, flux_mask

def main():
    # Panorama paths
    image_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/testOutput/panorama_output/pano.png"
    mask_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/testOutput/panorama_output/mask_for_inpainting.png"
    output_path = "flux_inpainted_panorama.png"
    
    # Panorama-specific prompt
    prompt = "complete urban street panorama with buildings, infrastructure, and natural environment"
    
    print("🌍 FLUX Panorama Inpainting")
    print(f"Input: {image_path}")
    print(f"Mask: {mask_path}")
    print(f"Output: {output_path}")
    print(f"Prompt: {prompt}")
    
    try:
        # Run panorama inpainting
        result, original, flux_mask = flux_inpaint_panorama(image_path, mask_path, prompt, output_path)
        
        # Save result
        result.save(output_path)
        
        # Create comparison for panorama
        # For panorama, we create a vertical comparison due to wide aspect ratio
        original_small = original.resize((1280, 640), Image.LANCZOS)
        result_small = result.resize((1280, 640), Image.LANCZOS)
        
        # Convert flux_mask for visualization (white=inpaint areas)
        mask_vis = cv2.cvtColor(flux_mask, cv2.COLOR_GRAY2RGB)
        mask_small = Image.fromarray(mask_vis).resize((1280, 640), Image.NEAREST)
        
        # Create vertical comparison
        comparison_height = 640 * 3
        comparison = Image.new('RGB', (1280, comparison_height))
        comparison.paste(original_small, (0, 0))
        comparison.paste(mask_small, (0, 640))
        comparison.paste(result_small, (0, 1280))
        
        comparison.save("flux_inpaint_panorama_comparison.png")
        
        print(f"✅ Panorama inpainting completed successfully!")
        print(f"Result saved to: {output_path}")
        print(f"Comparison saved to: flux_inpaint_panorama_comparison.png")
        
    except Exception as e:
        print(f"❌ Panorama inpainting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()