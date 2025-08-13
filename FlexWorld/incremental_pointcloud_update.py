#!/usr/bin/env python3
"""
增量式点云RGBD补全
基于FLUX inpainting结果，逐步更新和扩展3D点云
"""

import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image

# Add FlexWorld and FLUX paths
sys.path.append('/home/liujiajun/HunyuanWorld-1.0/FlexWorld')
sys.path.append('/home/liujiajun/HunyuanWorld-1.0/FLUX-Controlnet-Inpainting')

# FlexWorld imports
from ops.PcdMgr import PcdMgr
from ops.cam_utils import CamPlanner, Mcam
from ops.utils.depth import depth2pcd_world, refine_depth2

# FLUX imports
from diffusers.utils import load_image
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

class IncrementalPointCloudUpdater:
    def __init__(self, device='cuda'):
        self.device = device
        self.flux_pipe = None
        self.load_flux_model()
        
    def load_flux_model(self):
        """加载FLUX模型 - 复用已验证的加载逻辑"""
        print("🎨 Loading FLUX models...")
        
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
        
        self.flux_pipe = FluxControlNetInpaintingPipeline.from_pretrained(
            "/mnt/pretrained/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        ).to(self.device)
        
        self.flux_pipe.transformer.to(torch.bfloat16)
        self.flux_pipe.controlnet.to(torch.bfloat16)
        print("✅ FLUX models loaded!")

    def flux_inpaint(self, rgb_image, alpha_mask, prompt="urban street scene"):
        """使用FLUX进行图像补全"""
        size = (768, 768)
        
        # 处理FlexWorld alpha mask (只有=0的像素才inpaint)
        flux_mask = np.zeros_like(alpha_mask)
        flux_mask[alpha_mask == 0] = 255
        
        # 转换为PIL并resize
        image_pil = Image.fromarray(rgb_image).resize(size, Image.LANCZOS)
        mask_pil = Image.fromarray(flux_mask).resize(size, Image.NEAREST)
        
        # FLUX inpainting
        generator = torch.Generator(device=self.device).manual_seed(42)
        result = self.flux_pipe(
            prompt=prompt,
            height=size[1], width=size[0],
            control_image=image_pil,
            control_mask=mask_pil,
            num_inference_steps=28,
            generator=generator,
            controlnet_conditioning_scale=0.9,
            guidance_scale=3.5,
            negative_prompt="blurry, low quality",
            true_guidance_scale=1.0
        ).images[0]
        
        # 转回512×512
        return np.array(result.resize((512, 512), Image.LANCZOS))
    
    def estimate_depth_simple(self, rgb_image):
        """简单深度估计 (基于梯度)"""
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # 计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 梯度 → 深度 (高梯度=边缘=近，低梯度=平坦=远)
        gradient_norm = gradient_mag / (np.max(gradient_mag) + 1e-8)
        depth_map = 1.0 - gradient_norm
        
        # 平滑并缩放到合理深度范围
        depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
        depth_map = 0.5 + depth_map * 4.5  # [0.5, 5.0]
        
        return depth_map
    
    def update_pointcloud_with_rgbd(self, pcd: PcdMgr, completed_rgb, estimated_depth, 
                                  cam: Mcam, alpha_mask):
        """将补全的RGBD数据更新到点云中"""
        # 只添加原本缺失(alpha=0)的像素点
        missing_mask = alpha_mask == 0
        
        if np.sum(missing_mask) == 0:
            print("  No missing pixels to add")
            return pcd
        
        # 转换RGBD为3D点
        new_points_3d = depth2pcd_world(estimated_depth, cam)  # [H, W, 3]
        
        # 准备RGB颜色 (归一化到[0,1])
        rgb_normalized = completed_rgb / 255.0  # [H, W, 3]
        
        # 只取缺失区域的点
        missing_pts_3d = new_points_3d[missing_mask]  # [N, 3]
        missing_rgb = rgb_normalized[missing_mask]    # [N, 3]
        
        # 合并为6D点 [N, 6] (XYZ + RGB)
        new_points_6d = np.concatenate([missing_pts_3d, missing_rgb], axis=1)
        
        print(f"  Adding {len(new_points_6d)} new points to pointcloud")
        
        # 添加到现有点云
        pcd.add_pts(new_points_6d)
        
        return pcd
    
    def process_single_view(self, pcd: PcdMgr, cam: Mcam, 
                           prompt="urban street scene", save_debug=False):
        """处理单个视角的增量补全"""
        print(f"\n=== Processing view ===")
        
        # 1. 渲染当前视角
        rgb_render = pcd.render(cam).squeeze().detach().cpu().numpy()  # [3, H, W]
        alpha_render = pcd.render(cam, mask=True).squeeze().detach().cpu().numpy()  # [H, W]
        
        # 转换格式 [3, H, W] → [H, W, 3]
        rgb_image = (rgb_render.transpose(1, 2, 0) * 255).astype(np.uint8)
        alpha_mask = (alpha_render * 255).astype(np.uint8)
        
        missing_pixels = np.sum(alpha_mask == 0)
        print(f"  Missing pixels: {missing_pixels} ({missing_pixels/np.prod(alpha_mask.shape)*100:.2f}%)")
        
        if missing_pixels == 0:
            print("  No missing pixels, skipping...")
            return pcd
        
        # 2. FLUX补全RGB
        print(f"  Running FLUX inpainting...")
        completed_rgb = self.flux_inpaint(rgb_image, alpha_mask, prompt)
        
        # 3. 估计深度
        print(f"  Estimating depth...")
        estimated_depth = self.estimate_depth_simple(completed_rgb)
        
        # 4. 更新点云
        print(f"  Updating pointcloud...")
        pcd = self.update_pointcloud_with_rgbd(pcd, completed_rgb, estimated_depth, cam, alpha_mask)
        
        # 5. 保存调试信息
        if save_debug:
            debug_dir = "incremental_debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            cv2.imwrite(f"{debug_dir}/original_rgb.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{debug_dir}/alpha_mask.png", alpha_mask)
            cv2.imwrite(f"{debug_dir}/completed_rgb.png", cv2.cvtColor(completed_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{debug_dir}/estimated_depth.png", (estimated_depth/estimated_depth.max()*255).astype(np.uint8))
            
            # 对比图
            comparison = np.hstack([rgb_image, completed_rgb])
            cv2.imwrite(f"{debug_dir}/comparison.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        return pcd
    
    def run_incremental_completion(self, pcd_path: str, num_views=12, 
                                  output_dir="incremental_output", 
                                  prompt="urban street scene"):
        """运行增量式点云补全"""
        print(f"🚀 Starting incremental pointcloud completion")
        print(f"  Input pointcloud: {pcd_path}")
        print(f"  Views to process: {num_views}")
        print(f"  Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 加载初始点云
        print(f"\n📂 Loading initial pointcloud...")
        pcd = PcdMgr(ply_file_path=pcd_path)
        initial_points = len(pcd.pts)
        print(f"  Initial points: {initial_points:,}")
        
        # 2. 创建相机轨迹 (360度环绕)
        print(f"\n📹 Creating camera trajectory...")
        planner = CamPlanner()
        traj = planner.add_traj().move_orbit_to(0, -360, 0.5, num_frames=num_views).finish()
        
        # 设置相机尺寸
        for cam in traj:
            cam.set_size(512, 512)
        
        print(f"  Created {len(traj)} camera poses")
        
        # 3. 逐视角处理
        for i, cam in enumerate(traj):
            print(f"\n--- View {i+1}/{len(traj)} ---")
            
            save_debug = (i == 0)  # 只保存第一个视角的调试信息
            pcd = self.process_single_view(pcd, cam, prompt, save_debug)
            
            # 每5个视角保存一次中间结果
            if (i + 1) % 5 == 0:
                intermediate_path = f"{output_dir}/pointcloud_after_{i+1}_views.ply"
                pcd.save_ply(intermediate_path)
                print(f"  💾 Saved intermediate result: {intermediate_path}")
        
        # 4. 保存最终结果
        final_points = len(pcd.pts)
        final_path = f"{output_dir}/completed_pointcloud.ply"
        pcd.save_ply(final_path)
        
        print(f"\n🎉 Incremental completion finished!")
        print(f"  Initial points: {initial_points:,}")
        print(f"  Final points: {final_points:,}")
        print(f"  Added points: {final_points - initial_points:,}")
        print(f"  Final pointcloud: {final_path}")
        
        return pcd

def main():
    # 配置
    pcd_path = "street_pointcloud.ply"
    output_dir = "incremental_street_output"
    prompt = "urban street scene with buildings, vehicles, and infrastructure"
    num_views = 12  # 从少量视角开始测试
    
    # 运行增量补全
    updater = IncrementalPointCloudUpdater()
    final_pcd = updater.run_incremental_completion(
        pcd_path, num_views, output_dir, prompt
    )
    
    print(f"\n✅ All done! Check results in {output_dir}/")

if __name__ == "__main__":
    main()