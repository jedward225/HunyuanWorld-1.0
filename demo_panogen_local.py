# Tencent HunyuanWorld-1.0 is licensed under TENCENT HUNYUANWORLD-1.0 COMMUNITY LICENSE AGREEMENT
# THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA AND 
# IS EXPRESSLY LIMITED TO THE TERRITORY, AS DEFINED BELOW.
# By clicking to agree or by using, reproducing, modifying, distributing, performing or displaying 
# any portion or element of the Tencent HunyuanWorld-1.0 Works, including via any Hosted Service, 
# You will be deemed to have recognized and accepted the content of this Agreement, 
# which is effective immediately.

# For avoidance of doubts, Tencent HunyuanWorld-1.0 means the 3D generation models 
# and their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent at [https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0].
import os
import torch
import numpy as np 

import cv2
from PIL import Image

import argparse

# huanyuan3d text to panorama
from hy3dworld import Text2PanoramaPipelines

# huanyuan3d image to panorama
from hy3dworld import Image2PanoramaPipelines
from hy3dworld import Perspective


class Text2PanoramaDemo:
    def __init__(self, use_local_model=True):
        # set default parameters
        self.height = 960
        self.width = 1920

        # panorama parameters
        # these parameters are used to control the panorama generation
        # you can adjust them according to your needs
        self.guidance_scale = 30
        self.shifting_extend = 0
        self.num_inference_steps = 50
        self.true_cfg_scale = 0.0
        self.blend_extend = 6

        # model paths
        if use_local_model:
            # Use local FLUX.1-dev model when available
            self.model_path = "/mnt/pretrained/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
            self.lora_path = "/mnt/zhouzihan/HunyuanWorld-1.0/models"
        else:
            self.lora_path = "tencent/HunyuanWorld-1"
            self.model_path = "black-forest-labs/FLUX.1-dev"
        
        print(f"Using model path: {self.model_path}")
        print(f"Using LoRA path: {self.lora_path}")
        
        # load the pipeline
        # use bfloat16 to save some VRAM
        self.pipe = Text2PanoramaPipelines.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=use_local_model
        ).to("cuda")
        
        # and enable lora weights
        if use_local_model:
            lora_weight_path = os.path.join(self.lora_path, "HunyuanWorld-PanoDiT-Text", "lora.safetensors")
            self.pipe.load_lora_weights(
                lora_weight_path,
                torch_dtype=torch.bfloat16
            )
        else:
            self.pipe.load_lora_weights(
                self.lora_path,
                subfolder="HunyuanWorld-PanoDiT-Text",
                weight_name="lora.safetensors",
                torch_dtype=torch.bfloat16
            )
        
        # save some VRAM by offloading the model to CPU
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()  # and enable vae tiling to save some VRAM

    def run(self, prompt, negative_prompt=None, seed=42, output_path='output_panorama'):
        # get panorama
        image = self.pipe(
            prompt,
            height=self.height,
            width=self.width,
            negative_prompt=negative_prompt,
            generator=torch.Generator("cpu").manual_seed(seed),
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            blend_extend=self.blend_extend,
            true_cfg_scale=self.true_cfg_scale,
        ).images[0]

        # create output directory if it does not exist
        os.makedirs(output_path, exist_ok=True)
        # save the panorama image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        # save the image to the output path
        image.save(os.path.join(output_path, 'panorama.png'))

        return image


class Image2PanoramaDemo:
    def __init__(self, use_local_model=True):
        # set default parameters
        self.height, self.width = 960, 1920  # 768, 1536 #

        # panorama parameters
        # these parameters are used to control the panorama generation
        # you can adjust them according to your needs
        self.THETA = 0
        self.PHI = 0
        self.guide_ratio = 0.9
        self.FOV = 67.5
        self.res_scale = 0.25
        self.p2p_scale = 2

        self.guidance_scale = 20  # 30
        self.true_cfg_scale = 0.0
        self.shifting_extend = 0
        self.blend_extend = 7
        self.num_inference_steps = 50

        # model paths
        if use_local_model:
            # Use local FLUX.1-dev model when available
            self.model_path = "/mnt/pretrained/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
            self.lora_path = "/mnt/zhouzihan/HunyuanWorld-1.0/models"
        else:
            self.lora_path = "tencent/HunyuanWorld-1"
            self.model_path = "black-forest-labs/FLUX.1-dev"
        
        print(f"Using model path: {self.model_path}")
        print(f"Using LoRA path: {self.lora_path}")

        # use bfloat16 to save some VRAM
        self.pipe = Image2PanoramaPipelines.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=use_local_model
        ).to("cuda")
        
        # and enable lora weights
        if use_local_model:
            lora_weight_path = os.path.join(self.lora_path, "HunyuanWorld-PanoDiT-Image", "lora.safetensors")
            self.pipe.load_lora_weights(
                lora_weight_path,
                torch_dtype=torch.bfloat16
            )
        else:
            self.pipe.load_lora_weights(
                self.lora_path,
                subfolder="HunyuanWorld-PanoDiT-Image",
                weight_name="lora.safetensors",
                torch_dtype=torch.bfloat16
            )
        
        # save some VRAM by offloading the model to CPU
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()

    def run(self, image_path=None, output_path='output_panorama', seed=42):
        p2p = Perspective()

        if image_path and os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            raise ValueError(f"Image path {image_path} does not exist.")

        # get panorama
        panorama_image = self.pipe(
            image,
            p2p,
            height=self.height,
            width=self.width,
            THETA=self.THETA,
            PHI=self.PHI,
            FOV=self.FOV,
            res_scale=self.res_scale,
            guide_ratio=self.guide_ratio,
            p2p_scale=self.p2p_scale,
            generator=torch.Generator("cpu").manual_seed(seed),
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            blend_extend=self.blend_extend,
            shifting_extend=self.shifting_extend,
            true_cfg_scale=self.true_cfg_scale,
        )
        # create output directory if it does not exist
        os.makedirs(output_path, exist_ok=True)
        # save the panorama image
        panorama_image.save(os.path.join(output_path, 'panorama.png'))

        return panorama_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate panoramic image with Tencent HY3d World 1.0')
    parser.add_argument('--prompt', type=str, default='', help='Text prompt for panorama generation')
    parser.add_argument('--negative_prompt', type=str, default=None, help='Negative text prompt for panorama generation')
    parser.add_argument('--image_path', type=str, default=None, help='Path to input image for image-to-panorama generation')
    parser.add_argument('--output_path', type=str, default='output', help='Output directory path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
    parser.add_argument('--use_local', action='store_true', default=True, help='Use local models instead of downloading')
    
    args = parser.parse_args()
    
    # create output directory if it does not exist
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"Output will be saved to: {args.output_path}")
    
    # Check if model exists
    model_path = "/mnt/pretrained/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
    if args.use_local and not os.path.exists(model_path):
        print(f"Warning: Local model path {model_path} does not exist yet.")
        print("Please wait for the FLUX.1-dev model to be copied to /mnt/pretrained/")
        print("Or use --use_local=False to download from HuggingFace (requires access)")
        exit(1)
    
    if args.image_path:
        print(f"Using image-to-panorama generation with image: {args.image_path}")
        demo_I2P = Image2PanoramaDemo(use_local_model=args.use_local)
        panorama_image = demo_I2P.run(
            image_path=args.image_path,
            output_path=args.output_path,
            seed=args.seed
        )
    else:
        print("No image path provided, using text-to-panorama generation.")
        demo_T2P = Text2PanoramaDemo(use_local_model=args.use_local)
        panorama_image = demo_T2P.run(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            output_path=args.output_path
        )
    
    print(f"Panorama saved to: {os.path.join(args.output_path, 'panorama.png')}")