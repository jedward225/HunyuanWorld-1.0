
import sys
sys.path.append('/home/liujiajun/HunyuanWorld-1.0/FLUX-Controlnet-Inpainting')
import torch
import numpy as np
from PIL import Image
import cv2

from diffusers.utils import load_image
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

# 加载图像和mask
rgb_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/realOutput/frames/frame_001.png"
mask_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/realOutput/frames/mask_001.png"

rgb_img = cv2.imread(rgb_path)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 如果没有需要补全的区域，直接复制
if np.sum(mask) == 0:
    output_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/realOutput/inpainted/inpainted_001.png"
    cv2.imwrite(output_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    print("No inpainting needed")
else:
    # 初始化FLUX pipeline
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
    
    # 准备输入（768x768）
    size = (768, 768)
    image_pil = Image.fromarray(rgb_img).resize(size, Image.LANCZOS)
    mask_pil = Image.fromarray(mask).resize(size, Image.NEAREST)
    
    # 生成prompt
    avg_brightness = np.mean(rgb_img)
    if avg_brightness > 150:
        prompt = "complete urban street scene with buildings, bright daylight, photorealistic"
    else:
        prompt = "complete urban street scene with buildings, natural lighting, photorealistic"
    
    # FLUX推理
    generator = torch.Generator(device="cuda").manual_seed(42 + 1)
    
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
    output_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/realOutput/inpainted/inpainted_001.png"
    result_512.save(output_path)
    print(f"Inpainted and saved to {output_path}")
