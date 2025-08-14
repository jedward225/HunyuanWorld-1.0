import os
import sys
import subprocess
import json
import numpy as np
from datetime import datetime
import shutil
from pathlib import Path

class IncrementalPipeline:
    def __init__(self, pointcloud_path, output_dir=None, overwrite_pointclouds=True, max_frames=5):
        frame_size = 512
        fov = 67.5
        f = frame_size / (2 * np.tan(np.radians(fov/2)))
        # print(f"📷 Camera settings: FOV={fov}°, focal_length={f:.2f}, frame_size={frame_size}x{frame_size}")
        
        from ops.cam_utils import Mcam
        from ops.PcdMgr import PcdMgr
        Mcam.set_default_f(f)
        PcdMgr.set_default_render_backend('gs')
        self.pointcloud_path = pointcloud_path
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.overwrite_pointclouds = overwrite_pointclouds
        self.max_frames = max_frames
        
        if output_dir is None:
            output_dir = f"/home/liujiajun/HunyuanWorld-1.0/FlexWorld/realOutput"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建工作目录
        self.frames_dir = self.output_dir / "frames"
        self.inpainted_dir = self.output_dir / "inpainted"
        self.pointclouds_dir = self.output_dir / "pointclouds"
        self.frames_dir.mkdir(exist_ok=True)
        self.inpainted_dir.mkdir(exist_ok=True)
        self.pointclouds_dir.mkdir(exist_ok=True)
        
        # 检查原始点云大小
        pointcloud_size = os.path.getsize(pointcloud_path) / (1024 * 1024)  # MB
        print(f"📊 Original pointcloud size: {pointcloud_size:.1f} MB")
        
        if self.overwrite_pointclouds:
            # 覆盖式保存：只保留当前和备份
            self.current_pointcloud = self.pointclouds_dir / "pointcloud_current.ply"
            self.backup_pointcloud = self.pointclouds_dir / "pointcloud_backup.ply"
            shutil.copy(pointcloud_path, self.current_pointcloud)
            print(f"💾 Using overwrite mode (saves disk space)")
        else:
            # 完整保存：保留所有版本
            self.current_pointcloud = self.pointclouds_dir / "pointcloud_000.ply"
            shutil.copy(pointcloud_path, self.current_pointcloud)
            print(f"💾 Using full save mode (keeps all versions)")
        
        # 应用坐标系变换（只在初始化时执行一次）
        self._apply_coordinate_transform()
        
        # 初始化完整相机轨迹（预构建72帧轨迹）
        print("🎬 Building camera trajectory...")
        from ops.cam_utils import CamPlanner
        plan = CamPlanner()
        self.full_trajectory = plan.add_traj().move_orbit_to(0, -360, 0.5, num_frames=72).finish()
        for i in range(len(self.full_trajectory)):
            self.full_trajectory[i].set_size(512, 512)
        print(f"   Built trajectory with {len(self.full_trajectory)} cameras")
        
        print(f"📁 Pipeline initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Initial pointcloud: {self.current_pointcloud}")
        print(f"   Max frames to process: {self.max_frames}")
    
    def _apply_coordinate_transform(self):
        """
        应用坐标系变换（对齐ljj.py），只在初始化时执行一次
        """
        print("🔄 Applying coordinate transformation to initial pointcloud...")
        
        cmd = f"""
        bash -c "source /home/liujiajun/miniconda3/etc/profile.d/conda.sh && \
        conda activate HunyuanWorld && \
        cd /home/liujiajun/HunyuanWorld-1.0/FlexWorld && \
        python scripts/coordinate_transform_fixed.py \
        --input {self.current_pointcloud} \
        --output {self.current_pointcloud} \
        --angle_x 90 --angle_y -90 --angle_z 0"
        """
        
        print("🔧 Applying coordinate transformation")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
        
        if result.returncode != 0:
            print(f"❌ Error in coordinate transformation: {result.stderr}")
            print(f"❌ Output: {result.stdout}")
            raise RuntimeError(f"Failed to apply coordinate transformation")
        
        print(result.stdout.strip())
    
    def run_in_hunyuan_env(self, script_content, description="Running in HunyuanWorld"):
        """
        在HunyuanWorld环境中运行Python代码
        """
        script_path = self.output_dir / "temp_hunyuan_script.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        cmd = f"""
        bash -c "source /home/liujiajun/miniconda3/etc/profile.d/conda.sh && \
        conda activate HunyuanWorld && \
        cd /home/liujiajun/HunyuanWorld-1.0/FlexWorld && \
        python {script_path}"
        """
        
        print(f"🔧 {description}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
        
        if result.returncode != 0:
            print(f"❌ Error in HunyuanWorld: {result.stderr}")
            print(f"❌ Output: {result.stdout}")
            raise RuntimeError(f"Failed to run in HunyuanWorld environment")
        
        return result.stdout
    
    def run_in_flux_env(self, script_content, description="Running in flux-inpainting"):
        """
        在flux-inpainting环境中运行Python代码
        """
        script_path = self.output_dir / "temp_flux_script.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 修复shell兼容性问题
        cmd = f"""
        bash -c "source /home/liujiajun/miniconda3/etc/profile.d/conda.sh && \
        conda activate flux-inpainting && \
        cd /home/liujiajun/HunyuanWorld-1.0/FlexWorld && \
        python {script_path}"
        """
        
        print(f"🎨 {description}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
        
        if result.returncode != 0:
            print(f"❌ Error in flux-inpainting: {result.stderr}")
            print(f"❌ Output: {result.stdout}")
            raise RuntimeError(f"Failed to run in flux-inpainting environment")
        
        return result.stdout
    
    def render_frame(self, frame_idx, pointcloud_path, cam_data):
        """
        在HunyuanWorld环境中渲染单帧
        """
        script = f'''
import sys
sys.path.append('/home/liujiajun/HunyuanWorld-1.0/FlexWorld')
import numpy as np
import cv2
from ops.PcdMgr import PcdMgr
from ops.cam_utils import CamPlanner
import einops
import json

def rotate_point_cloud(point_cloud, angle_x_deg=0, angle_y_deg=0, angle_z_deg=0):
    """绕坐标系轴旋转点云"""
    angle_x_rad = np.deg2rad(angle_x_deg)
    angle_y_rad = np.deg2rad(angle_y_deg)
    angle_z_rad = np.deg2rad(angle_z_deg)
    
    cos_x, sin_x = np.cos(angle_x_rad), np.sin(angle_x_rad)
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])
    
    cos_y, sin_y = np.cos(angle_y_rad), np.sin(angle_y_rad)
    rotation_matrix_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])
    
    cos_z, sin_z = np.cos(angle_z_rad), np.sin(angle_z_rad)
    rotation_matrix_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])
    
    combined_rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x
    rotated_point_cloud = (combined_rotation_matrix @ point_cloud.T).T
    return rotated_point_cloud

# 加载点云（坐标系变换只在初始化时应用）
pcd = PcdMgr(ply_file_path="{pointcloud_path}")

# 重建相机对象（使用传入的相机数据）
from ops.cam_utils import Mcam
cam = Mcam()
cam.R = np.array({cam_data['R']})
cam.T = np.array({cam_data['T']})
cam.f = {cam_data['f']}
cam.c = np.array({cam_data['c']})
cam.set_size(512, 512)

# 渲染（使用gs后端，对齐ljj.py）
rgb = pcd.render(cam, backends="gs")  # 使用Gaussian Splatting后端
alpha = pcd.render(cam, mask=True, backends="gs")
depth = pcd.render(cam, depth=True, backends="gs")

# 转换格式并保存
rgb_img = einops.rearrange(rgb[0], 'c h w -> h w c').cpu().numpy()
rgb_img = (rgb_img * 255).astype(np.uint8)
alpha_img = (alpha[0,0].cpu().numpy() * 255).astype(np.uint8)
depth_img = depth[0,0].cpu().numpy()

# 保存结果
cv2.imwrite("{self.frames_dir}/frame_{frame_idx:03d}.png", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
cv2.imwrite("{self.frames_dir}/alpha_{frame_idx:03d}.png", alpha_img)
np.save("{self.frames_dir}/depth_{frame_idx:03d}.npy", depth_img)

# 生成mask（对齐flux_inpaint_simple.py的处理）
threshold = 6  # 与flux_inpaint_simple.py保持一致
mask = alpha_img <= threshold  # 低覆盖度像素需要inpaint

# 应用膨胀来扩展mask覆盖（对齐flux_inpaint_simple.py）
kernel = np.ones((3,3), np.uint8)
mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)  # 扩展mask约10像素

cv2.imwrite("{self.frames_dir}/mask_{frame_idx:03d}.png", mask_dilated * 255)

# 输出统计（使用膨胀后的mask）
stats = {{
    "missing_pixels": int(np.sum(mask_dilated)),
    "total_pixels": 512 * 512,
    "missing_ratio": float(np.sum(mask_dilated) / (512 * 512))
}}

print(json.dumps(stats))
'''
        
        output = self.run_in_hunyuan_env(script, f"Rendering frame {frame_idx:03d}")
        stats = json.loads(output.strip().split('\n')[-1])
        return stats
    
    def inpaint_frame(self, frame_idx):
        """
        在flux-inpainting环境中进行RGB补全
        """
        script = f'''
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
rgb_path = "{self.frames_dir}/frame_{frame_idx:03d}.png"
mask_path = "{self.frames_dir}/mask_{frame_idx:03d}.png"

rgb_img = cv2.imread(rgb_path)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 如果没有需要补全的区域，直接复制
if np.sum(mask) == 0:
    output_path = "{self.inpainted_dir}/inpainted_{frame_idx:03d}.png"
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
    generator = torch.Generator(device="cuda").manual_seed(42 + {frame_idx})
    
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
    output_path = "{self.inpainted_dir}/inpainted_{frame_idx:03d}.png"
    result_512.save(output_path)
    print(f"Inpainted and saved to {{output_path}}")
'''
        
        self.run_in_flux_env(script, f"Inpainting frame {frame_idx:03d}")
    
    def update_pointcloud(self, frame_idx, cam_data):
        """
        在HunyuanWorld环境中更新点云 - 带详细调试信息版本
        """
        script = f'''
import sys
sys.path.append('/home/liujiajun/HunyuanWorld-1.0/FlexWorld')
import numpy as np
import cv2
import open3d as o3d
from ops.PcdMgr import PcdMgr
from ops.cam_utils import CamPlanner
from ops.utils.depth import refine_depth2, depth2pcd_world
import os

def rotate_point_cloud(point_cloud, angle_x_deg=0, angle_y_deg=0, angle_z_deg=0):
    """绕坐标系轴旋转点云"""
    angle_x_rad = np.deg2rad(angle_x_deg)
    angle_y_rad = np.deg2rad(angle_y_deg)
    angle_z_rad = np.deg2rad(angle_z_deg)
    
    cos_x, sin_x = np.cos(angle_x_rad), np.sin(angle_x_rad)
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])
    
    cos_y, sin_y = np.cos(angle_y_rad), np.sin(angle_y_rad)
    rotation_matrix_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])
    
    cos_z, sin_z = np.cos(angle_z_rad), np.sin(angle_z_rad)
    rotation_matrix_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])
    
    combined_rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x
    rotated_point_cloud = (combined_rotation_matrix @ point_cloud.T).T
    return rotated_point_cloud

print(f"\\n🔍 ===== FRAME {frame_idx:03d} DEBUG INFO =====")

# 加载当前点云
current_pcd = PcdMgr(ply_file_path="{self.current_pointcloud}")
original_point_count = len(current_pcd.pts)
pointcloud_size_mb = os.path.getsize("{self.current_pointcloud}") / (1024 * 1024)
print(f"📊 Original pointcloud: {{original_point_count}} points, {{pointcloud_size_mb:.1f}} MB")

# 加载数据
inpainted_rgb = cv2.imread("{self.inpainted_dir}/inpainted_{frame_idx:03d}.png")
inpainted_rgb = cv2.cvtColor(inpainted_rgb, cv2.COLOR_BGR2RGB)
original_depth = np.load("{self.frames_dir}/depth_{frame_idx:03d}.npy")
mask = cv2.imread("{self.frames_dir}/mask_{frame_idx:03d}.png", cv2.IMREAD_GRAYSCALE) > 0

print(f"📷 Depth range: [{{original_depth.min():.3f}}, {{original_depth.max():.3f}}]")
print(f"🎭 Mask coverage: {{np.sum(mask)}} / {{mask.size}} pixels ({{np.sum(mask)/mask.size*100:.2f}}%)")

# 重建相机对象（与渲染时完全相同）
from ops.cam_utils import Mcam
cam = Mcam()
cam.R = np.array({cam_data['R']})
cam.T = np.array({cam_data['T']})
cam.f = {cam_data['f']}
cam.c = np.array({cam_data['c']})
cam.set_size(512, 512)

print(f"📹 Camera: f={{cam.f}}, c={{cam.c}}")
print(f"📍 Camera pos: [{{cam.T[0]:.3f}}, {{cam.T[1]:.3f}}, {{cam.T[2]:.3f}}]")

if np.sum(mask) > 0:
    # 估计深度（简单插值方法）
    estimated_depth = cv2.inpaint(
        original_depth.astype(np.float32),
        mask.astype(np.uint8) * 255,
        inpaintRadius=10,
        flags=cv2.INPAINT_TELEA
    )
    
    print(f"🔧 Estimated depth range: [{{estimated_depth.min():.3f}}, {{estimated_depth.max():.3f}}]")
    
    # 深度对齐
    aligned_depth = refine_depth2(
        render_dpt=original_depth,
        ipaint_dpt=estimated_depth,
        ipaint_msk=mask,
        iters=50,
        blur_size=15,
        scaled=True
    )
    
    print(f"⚖️  Aligned depth range: [{{aligned_depth.min():.3f}}, {{aligned_depth.max():.3f}}]")
    
    # 3D重建
    points_3d = depth2pcd_world(aligned_depth, cam)
    
    # 只添加mask区域的点
    new_points_3d = points_3d[mask]
    new_colors = inpainted_rgb[mask] / 255.0
    new_points_6d = np.concatenate([new_points_3d, new_colors], axis=-1)
    
    print(f"🎯 New points generated: {{len(new_points_6d)}}")
    
    if len(new_points_6d) > 0:
        # 分析新增点的空间分布
        x_range = [new_points_3d[:, 0].min(), new_points_3d[:, 0].max()]
        y_range = [new_points_3d[:, 1].min(), new_points_3d[:, 1].max()]
        z_range = [new_points_3d[:, 2].min(), new_points_3d[:, 2].max()]
        
        print(f"📐 New points X: [{{x_range[0]:.3f}}, {{x_range[1]:.3f}}]")
        print(f"📐 New points Y: [{{y_range[0]:.3f}}, {{y_range[1]:.3f}}]")
        print(f"📐 New points Z: [{{z_range[0]:.3f}}, {{z_range[1]:.3f}}]")
        
        # ⚠️ 移除离群点过滤，直接添加
        print("⚠️  SKIPPING outlier removal for debugging!")
        current_pcd.add_pts(new_points_6d)
        points_added = len(new_points_6d)
        
        print(f"✅ Added {{points_added}} points directly (no filtering)")
    else:
        points_added = 0
        print("❌ No points to add")
else:
    points_added = 0
    print("⏭️  No missing pixels, skipping depth estimation")

# 保存更新后的点云（根据保存模式）
if {self.overwrite_pointclouds}:
    output_path = "{self.current_pointcloud}"
else:
    output_path = "{self.pointclouds_dir}/pointcloud_{frame_idx+1:03d}.ply"

pts = current_pcd.pts
final_point_count = len(pts)

# 检查是否有法向量数据（PcdMgr可能包含法向量）
if pts.shape[1] >= 9:  # x,y,z,nx,ny,nz,r,g,b
    print("💾 保存点云（包含法向量）")
    # 使用原始文件格式保存法向量
    # 读取原始文件格式作为模板
    template_pcd = o3d.io.read_point_cloud("{self.pointcloud_path}")
    if template_pcd.has_normals():
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pts[:, :3])
        pcd_o3d.normals = o3d.utility.Vector3dVector(pts[:, 3:6])  # 假设法向量在3:6
        pcd_o3d.colors = o3d.utility.Vector3dVector(pts[:, 6:9])   # 颜色在6:9
        o3d.io.write_point_cloud(output_path, pcd_o3d)
    else:
        # 没有法向量模板，使用标准格式
        pcd_o3d = o3d.geometry.PointCloud() 
        pcd_o3d.points = o3d.utility.Vector3dVector(pts[:, :3])
        pcd_o3d.colors = o3d.utility.Vector3dVector(pts[:, 3:6])
        o3d.io.write_point_cloud(output_path, pcd_o3d)
else:
    # 标准格式：x,y,z,r,g,b
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(pts[:, 3:6])
    o3d.io.write_point_cloud(output_path, pcd_o3d)

# 计算文件大小
final_size_mb = os.path.getsize(output_path) / (1024 * 1024)
size_change = final_size_mb - pointcloud_size_mb

print(f"💾 Final pointcloud: {{final_point_count}} points, {{final_size_mb:.1f}} MB")
print(f"📈 Change: +{{final_point_count - original_point_count}} points, {{size_change:+.1f}} MB")
print(f"🎯 Points added this frame: {{points_added}}")
print(f"===== FRAME {frame_idx:03d} COMPLETED =====\\n")
'''
        
        self.run_in_hunyuan_env(script, f"Updating pointcloud after frame {frame_idx:03d}")
        
        # 更新当前点云路径（根据保存模式）
        if self.overwrite_pointclouds:
            # 覆盖模式：备份当前，更新current
            if self.current_pointcloud.exists():
                shutil.copy(self.current_pointcloud, self.backup_pointcloud)
            # current_pointcloud路径保持不变，内容已更新
        else:
            # 完整保存模式：每帧保存新文件
            self.current_pointcloud = self.pointclouds_dir / f"pointcloud_{frame_idx+1:03d}.ply"
    
    def run(self, num_frames=None):
        """
        运行完整的增量式pipeline
        """
        if num_frames is None:
            num_frames = self.max_frames
            
        print(f"\n🚀 Starting Incremental RGBD Completion Pipeline")
        print(f"📊 Processing {num_frames} frames")
        print("="*50)
        
        for i in range(num_frames):
            print(f"\n📷 Frame {i+1}/{num_frames}")
            print("-"*30)
            
            # 获取当前帧的相机（从预构建轨迹中）
            if i >= len(self.full_trajectory):
                print(f"❌ Frame {i} exceeds trajectory length {len(self.full_trajectory)}")
                break
                
            cam = self.full_trajectory[i]
            # 提取相机参数传递给子环境
            cam_data = {
                'R': cam.R.tolist(),
                'T': cam.T.tolist(), 
                'f': cam.f,
                'c': list(cam.c) if isinstance(cam.c, tuple) else cam.c.tolist()
            }
            print(f"   Camera position: frame {i}/{len(self.full_trajectory)}")
            
            # 1. 在HunyuanWorld环境中渲染
            stats = self.render_frame(i, self.current_pointcloud, cam_data)
            print(f"   Missing pixels: {stats['missing_pixels']} ({stats['missing_ratio']*100:.2f}%)")
            
            # 2. 在flux-inpainting环境中补全
            if stats['missing_pixels'] > 0:
                self.inpaint_frame(i)
                
                # 3. 在HunyuanWorld环境中更新点云
                self.update_pointcloud(i, cam_data)
            else:
                print("   ✅ No inpainting needed")
            
            # 保存每帧结果供调试
            self.save_frame_results(i)
        
        print("\n" + "="*50)
        print("✅ Pipeline completed successfully!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"🎯 Final pointcloud: {self.current_pointcloud}")
        
        return str(self.current_pointcloud)
    
    def save_frame_results(self, frame_idx):
        """
        保存每帧结果供调试
        """
        print(f"💾 Saving frame {frame_idx:03d} results...")
        # 文件已经在render_frame和inpaint_frame中保存了
        # 这里主要是提示信息
        
    def generate_visualization(self, frame_num):
        """
        生成中间结果的可视化
        """
        print(f"\n📊 Generating visualization at frame {frame_num}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Incremental RGBD Completion Pipeline')
    parser.add_argument('--pointcloud', type=str, 
                       default='/home/liujiajun/HunyuanWorld-1.0/FlexWorld/street_pointcloud_new.ply',
                       help='Input point cloud path')
    parser.add_argument('--frames', type=int, default=5,
                       help='Number of frames to process (default: 5 for testing)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--overwrite-pointclouds', action='store_true', default=True,
                       help='Use overwrite mode to save disk space (default: True)')
    parser.add_argument('--full-save', action='store_true', default=False,
                       help='Save all pointcloud versions (uses more disk space)')
    
    args = parser.parse_args()
    
    overwrite_mode = not args.full_save  # 如果指定full_save，则不使用overwrite
    
    print(f"🔧 Configuration:")
    print(f"   Pointcloud: {args.pointcloud}")
    print(f"   Frames: {args.frames}")
    print(f"   Save mode: {'Overwrite (disk-efficient)' if overwrite_mode else 'Full (keeps all versions)'}")
    
    # 创建pipeline
    pipeline = IncrementalPipeline(
        pointcloud_path=args.pointcloud, 
        output_dir=args.output,
        overwrite_pointclouds=overwrite_mode,
        max_frames=args.frames
    )
    
    final_pointcloud = pipeline.run(num_frames=args.frames)
    
    print(f"\n🎉 Success! Final pointcloud: {final_pointcloud}")
