#!/usr/bin/env python3
"""
Simple FLUX Controlnet Inpainting based on official example
Inpaints the missing/masked regions in the provided image
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

def create_mask_for_missing_regions(image_path):
    """
    Create a mask for the missing/incomplete regions in the image
    In this case, we'll create a mask for the lower portion that appears incomplete
    """
    # Load image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Create mask - white areas will be inpainted
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Mask the lower portion that looks incomplete/cut off
    mask[int(h*0.7):, :] = 255
    
    # Also mask some gaps/missing areas if detected
    # You can adjust this based on your specific image
    
    return mask

def flux_inpaint_image(image_path, mask=None, prompt="urban street scene"):
    """
    Inpaint image using FLUX Controlnet following official example
    """
    
    print("🎨 Loading FLUX models...")
    
    # Build pipeline exactly like official example
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
    
    # Load image and mask - 恢复768×768处理以获得最佳FLUX效果
    size = (768, 768)  # FLUX最优分辨率
    image = load_image(image_path).convert("RGB").resize(size, Image.LANCZOS)  # 512→768 LANCZOS插值
    
    if mask is None:
        # Create mask for missing regions
        mask_array = create_mask_for_missing_regions(image_path)
        mask = Image.fromarray(mask_array).convert("RGB")
    else:
        # Load FlexWorld alpha mask
        flexworld_mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        
        # FlexWorld mask processing: only pixels with value=0 need inpainting
        # Create FLUX mask: white(255)=inpaint, black(0)=keep
        flux_mask = np.zeros_like(flexworld_mask)
        threshold = 6  # Coverage threshold: pixels <= threshold need inpainting
        flux_mask[flexworld_mask <= threshold] = 255  # Low coverage pixels get inpainted
        
        # Apply dilation to expand mask coverage (following FlexWorld implementation)
        kernel = np.ones((3,3), np.uint8)
        flux_mask = cv2.dilate(flux_mask, kernel, iterations=2)  # Expand mask by ~10 pixels
        
        print(f"  FlexWorld mask analysis:")
        print(f"    Need inpaint (≤{threshold}): {np.sum(flexworld_mask <= threshold)} pixels")
        print(f"    After dilation: {np.sum(flux_mask > 0)} pixels") 
        print(f"    Has geometry (>{threshold}): {np.sum(flexworld_mask > threshold)} pixels")
        print(f"    Final inpaint ratio: {np.sum(flux_mask > 0)/np.prod(flexworld_mask.shape)*100:.2f}%")
        
        # Convert to PIL并resize到768×768
        mask = Image.fromarray(flux_mask).convert("RGB").resize(size, Image.NEAREST)  # mask用NEAREST避免插值模糊
    
    # Set random seed for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(42)
    
    print(f"🎨 Starting inpainting with prompt: '{prompt}'")
    
    # Inpaint - following official parameters
    result = pipe(
        prompt=prompt,
        height=size[1],
        width=size[0],
        control_image=image,
        control_mask=mask,
        # num_inference_steps=28,
        num_inference_steps=20,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        # negative_prompt="low quality, distorted",
        negative_prompt="",
        true_guidance_scale=1.0
    ).images[0]
    
    # 将结果resize回FlexWorld的512×512
    result_512 = result.resize((512, 512), Image.LANCZOS)
    
    return result_512, image, mask

def main():
    # Input image path (the provided street scene)
    image_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/testOutput/frames/frame_009.png"
    mask_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/testOutput/frames/mask_009.png"
    output_path = "flux_inpainted_result.png"
    
    # Prompt describing what should be inpainted
    # prompt = "complete urban street scene with buildings, road surface, and infrastructure details"
    prompt = "complete the missing parts naturally"
    
    print("🚀 FLUX Image Inpainting")
    print(f"Input: {image_path}")
    print(f"Mask: {mask_path}")
    print(f"Output: {output_path}")
    print(f"Prompt: {prompt}")
    
    try:
        # Run inpainting with the provided mask
        result, original, mask = flux_inpaint_image(image_path, mask=mask_path, prompt=prompt)
        
        # Save results
        result.save(output_path)
        
        # Save comparison - resize all images to 512×512 for consistent comparison
        original_512 = original.resize((512, 512), Image.LANCZOS)
        mask_512 = mask.resize((512, 512), Image.NEAREST)
        # result is already 512×512
        
        # Create side-by-side comparison
        comparison_width = 512 * 3
        comparison = Image.new('RGB', (comparison_width, 512))
        comparison.paste(original_512, (0, 0))
        comparison.paste(mask_512, (512, 0))
        comparison.paste(result, (1024, 0))
        
        comparison.save("flux_inpaint_comparison.png")
        
        print(f"✅ Inpainting completed successfully!")
        print(f"Result saved to: {output_path}")
        print(f"Comparison saved to: flux_inpaint_comparison.png")
        
    except Exception as e:
        print(f"❌ Inpainting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()