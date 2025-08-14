#!/usr/bin/env python3
"""
FLUX inpainting脚本
从incremental_pipeline.py中提取的inpaint_frame功能
"""

import sys
import argparse
import torch
import numpy as np
from PIL import Image
import cv2

sys.path.append('/home/liujiajun/HunyuanWorld-1.0/FLUX-Controlnet-Inpainting')
from diffusers.utils import load_image
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

def main():
    parser = argparse.ArgumentParser(description='FLUX inpainting for single frame')
    parser.add_argument('--rgb_path', required=True, help='Input RGB image path')
    parser.add_argument('--mask_path', required=True, help='Input mask path')
    parser.add_argument('--output_path', required=True, help='Output inpainted image path')
    parser.add_argument('--frame_idx', type=int, required=True, help='Frame index (for seed)')
    parser.add_argument('--controlnet_path', default='/mnt2/FLUX.1-dev-Controlnet-Inpainting-Alpha', 
                       help='ControlNet model path')
    parser.add_argument('--transformer_path', 
                       default='/mnt/pretrained/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21',
                       help='Transformer model path')
    parser.add_argument('--prompt', 
                       default='complete urban street scene with buildings, natural lighting, photorealistic',
                       help='Inpainting prompt')
    
    args = parser.parse_args()

    # 加载图像和mask
    rgb_img = cv2.imread(args.rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)

    # 如果没有需要补全的区域，直接复制
    if np.sum(mask) == 0:
        cv2.imwrite(args.output_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        print("No inpainting needed")
        return

    print(f"Inpainting {np.sum(mask)} pixels...")

    # 初始化FLUX pipeline
    controlnet = FluxControlNetModel.from_pretrained(
        args.controlnet_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    
    transformer = FluxTransformer2DModel.from_pretrained(
        args.transformer_path,
        subfolder='transformer',
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        args.transformer_path,
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to("cuda")
    
    # 准备输入（768x768）
    size = (768, 768)
    image_pil = Image.fromarray(rgb_img).resize(size, Image.LANCZOS)
    mask_pil = Image.fromarray(mask).resize(size, Image.NEAREST)
    
    # 自适应prompt
    avg_brightness = np.mean(rgb_img)
    if avg_brightness > 150:
        prompt = "complete urban street scene with buildings, bright daylight, photorealistic"
    else:
        prompt = args.prompt
    
    print(f"Using prompt: {prompt}")
    
    # FLUX推理
    generator = torch.Generator(device="cuda").manual_seed(42 + args.frame_idx)
    
    result = pipe(
        prompt=prompt,
        height=size[1],
        width=size[0],
        control_image=image_pil,
        control_mask=mask_pil,
        num_inference_steps=20,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=1.0
    ).images[0]
    
    # 缩放回512x512并保存
    result_512 = result.resize((512, 512), Image.LANCZOS)
    result_512.save(args.output_path)
    print(f"Inpainted and saved to {args.output_path}")

if __name__ == "__main__":
    main()