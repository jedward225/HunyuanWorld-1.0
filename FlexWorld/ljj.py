### TODO: image2pcd看下要不要集成，让一次python运行就可以得到渲染的结果
### 看下hunyuanworld里有没有焦距f的设置，这个可能需要对齐一下
### 试着跑一跑代码，看看能不能渲染出视频，探索不同的轨迹
### 我们现在做到的是，给定一个点云文件，渲染出视频，现在看看能不能把这个视频反过来转换回去成全景图（问问gpt?）
### sky的处理，sky怎么渲染（可能比较难，我也不太会，优先级低一点）
### inpainting怎么做，需不需要额外的训练，还是说training free能做到？


### 下次讨论，只要能做到一张全景图，给定轨迹后，得到一张incomplete的全景图就行了（以及对应的mask）

from omegaconf import OmegaConf
from pipe.img2pcd import Image2Pcd_Tool
from ops.PcdMgr import PcdMgr
from ops.cam_utils import Mcam, CamPlanner
import torch
import torch.nn.functional as F
from pipe.view_extend import save_video
# f=351.3848876953125 * 1.0
import numpy as np
import cv2
from PIL import Image
import os
from ops.utils.general import easy_save_video

# def image2pcd(img):
#     """
#     从图像生成点云
#     """
   
#     # xxx是一个[N,6]的numpy/tensor ，xyzrgb
#     # 也可以是一个ply路径
#     pcd =  PcdMgr(pts3d = xxx)
#     return pcd

def rotate_point_cloud(point_cloud, angle_x_deg=0, angle_y_deg=0, angle_z_deg=0):
    """
    绕坐标系轴旋转点云。

    参数:
    point_cloud (np.ndarray): 要旋转的点云，形状为 (N, 3)。
    angle_x_deg (float): 绕 X 轴的旋转角度（度）。
    angle_y_deg (float): 绕 Y 轴的旋转角度（度）。
    angle_z_deg (float): 绕 Z 轴的旋转角度（度）。

    返回:
    np.ndarray: 旋转后的点云，形状为 (N, 3)。
    """
    # 1. 将角度从度转换为弧度
    angle_x_rad = np.deg2rad(angle_x_deg)
    angle_y_rad = np.deg2rad(angle_y_deg)
    angle_z_rad = np.deg2rad(angle_z_deg)

    # 2. 创建绕 X 轴的旋转矩阵
    cos_x, sin_x = np.cos(angle_x_rad), np.sin(angle_x_rad)
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])

    # 3. 创建绕 Y 轴的旋转矩阵
    cos_y, sin_y = np.cos(angle_y_rad), np.sin(angle_y_rad)
    rotation_matrix_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])

    # 4. 创建绕 Z 轴的旋转矩阵
    cos_z, sin_z = np.cos(angle_z_rad), np.sin(angle_z_rad)
    rotation_matrix_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])

    # 5. 组合旋转矩阵 (顺序: Z -> Y -> X)
    #    p' = Rz * Ry * Rx * p
    combined_rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x

    # 6. 将旋转应用于点云
    #    对于形状为 (N, 3) 的点云，我们需要对每个点 p 进行 R * p 的操作。
    #    这可以通过 (R @ P.T).T 来高效完成。
    rotated_point_cloud = (combined_rotation_matrix @ point_cloud.T).T

    return rotated_point_cloud

def frames_to_panorama_advanced(frames, trajectory, output_width=1920, output_height=960):
    """
    从视频帧重建清晰的全景图，避免重叠透明效果。
    为每个全景图位置选择最佳的源帧，而不是混合所有帧。
    
    参数:
    frames: 视频帧列表 [N, H, W, C]
    trajectory: 相机轨迹（Mcam对象列表）
    output_width: 输出全景图宽度  
    output_height: 输出全景图高度
    
    返回:
    panorama: 清晰的全景图 numpy数组
    mask: 有效像素mask
    """
    if len(frames) == 0 or len(trajectory) == 0:
        return np.zeros((output_height, output_width, 3), dtype=np.uint8), np.zeros((output_height, output_width), dtype=np.uint8)
    
    if len(frames) != len(trajectory):
        print(f"Warning: frames ({len(frames)}) and trajectory ({len(trajectory)}) length mismatch")
        min_len = min(len(frames), len(trajectory))
        frames = frames[:min_len]
        trajectory = trajectory[:min_len]
    
    # 获取帧尺寸
    frame_h, frame_w = frames[0].shape[:2]
    
    # 初始化全景图和最佳帧选择
    panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    mask = np.zeros((output_height, output_width), dtype=np.uint8)
    best_frame_distance = np.full((output_height, output_width), np.inf)  # 用于选择最佳帧
    
    print(f"Reconstructing clean panorama from {len(frames)} frames...")
    
    # 为每个帧创建网格坐标
    y_coords, x_coords = np.meshgrid(np.arange(frame_h), np.arange(frame_w), indexing='ij')
    
    for idx, (frame, cam) in enumerate(zip(frames, trajectory)):
        # 获取相机参数
        K = cam.getK()
        R = cam.R
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # 计算当前相机的朝向中心（球面坐标）
        # 相机朝向是-Z方向
        cam_forward = -R[:, 2]  # 相机朝向向量
        cam_azimuth = np.arctan2(cam_forward[0], cam_forward[2])  # 相机朝向的方位角
        
        # 将像素坐标转换为归一化相机坐标
        x_norm = (x_coords - cx) / fx
        y_norm = (y_coords - cy) / fy
        
        # 创建方向向量（相机坐标系）
        dirs_cam = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=-1)
        dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
        
        # 转换到世界坐标系
        dirs_world = dirs_cam @ R.T
        
        # 计算球面坐标
        theta = np.arctan2(dirs_world[..., 0], dirs_world[..., 2])  # 方位角 [-π, π]
        phi = np.arcsin(np.clip(dirs_world[..., 1], -1, 1))  # 俯仰角 [-π/2, π/2]
        
        # 映射到全景图像素坐标
        pano_x = ((theta + np.pi) / (2 * np.pi) * output_width).astype(int)
        pano_y = ((phi + np.pi/2) / np.pi * output_height).astype(int)
        
        # 处理边界
        pano_x = np.clip(pano_x, 0, output_width - 1)
        pano_y = np.clip(pano_y, 0, output_height - 1)
        
        # 计算每个像素到相机朝向中心的角度距离
        pixel_azimuth = theta
        angle_distance = np.abs(((pixel_azimuth - cam_azimuth + np.pi) % (2 * np.pi)) - np.pi)
        
        # 只处理相机视野内的像素
        fov_h = 2 * np.arctan(frame_w / (2 * fx))
        fov_v = 2 * np.arctan(frame_h / (2 * fy))
        
        valid_h = angle_distance <= fov_h / 2
        valid_v = np.abs(phi) <= fov_v / 2
        valid_pixels = valid_h & valid_v
        
        # 对于每个有效像素，检查是否应该使用当前帧
        valid_y, valid_x = np.where(valid_pixels)
        
        for py, px in zip(valid_y, valid_x):
            pano_py, pano_px = pano_y[py, px], pano_x[py, px]
            current_distance = angle_distance[py, px]
            
            # 如果当前帧更接近该全景图位置的最优视角，则使用它
            if current_distance < best_frame_distance[pano_py, pano_px]:
                best_frame_distance[pano_py, pano_px] = current_distance
                panorama[pano_py, pano_px] = frame[py, px]
                mask[pano_py, pano_px] = 255
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(frames)} frames")
    
    # 计算覆盖率
    coverage = np.sum(mask > 0) / (output_height * output_width) * 100
    print(f"Clean panorama coverage: {coverage:.1f}%")
    
    return panorama, mask

def frames_to_panorama(frames, trajectory, output_width=1920, output_height=960):
    """
    将渲染的视频帧转换为全景图及其mask。
    
    参数:
    frames: 视频帧列表或numpy数组 [N, H, W, C]
    trajectory: 相机轨迹（Mcam对象列表）
    output_width: 输出全景图宽度
    output_height: 输出全景图高度
    
    返回:
    panorama: 全景图 numpy数组
    mask: 有效像素mask
    """
    return frames_to_panorama_advanced(frames, trajectory, output_width, output_height)

def render_panorama_from_trajectory(pcd, trajectory, output_dir='testOutput/panorama_output'):
    """
    从给定轨迹渲染全景图。
    
    参数:
    pcd: 点云对象
    trajectory: 相机轨迹
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 渲染视频 (先不保存，只获取帧)
    print(f"Rendering video frames...")
    render_results = CamPlanner._render_video(pcd, traj=trajectory, output_path=None)
    render_mask = CamPlanner._render_video(pcd, traj=trajectory, output_path=None, mask=True)
    
    # 使用easy_save_video保存视频
    video_path = os.path.join(output_dir, 'rendered_video.mp4')
    mask_video_path = os.path.join(output_dir, 'rendered_mask.mp4')
    
    # 转换为正确格式并保存
    frames_np = torch.stack(render_results).cpu().numpy()  # [T, H, W, C]
    
    # 处理mask - mask是2D的[H, W]，需要扩展到3通道
    mask_list = []
    for m in render_mask:
        if len(m.shape) == 2:  # [H, W]
            # 扩展到3通道
            m_3ch = m.unsqueeze(-1).expand(-1, -1, 3)  # [H, W, 3]
        elif len(m.shape) == 3:
            if m.shape[-1] == 1:  # [H, W, 1]
                m_3ch = m.expand(-1, -1, 3)  # [H, W, 3]
            else:  # already [H, W, 3]
                m_3ch = m
        else:
            print(f"Warning: unexpected mask shape {m.shape}")
            m_3ch = m
        mask_list.append(m_3ch)
    mask_frames_np = torch.stack(mask_list).cpu().numpy()  # [T, H, W, 3]
    
    easy_save_video(frames_np, video_path, fps=30, value_range="0,1")
    easy_save_video(mask_frames_np, mask_video_path, fps=30, value_range="0,1")
    
    print(f"Videos saved to {video_path} and {mask_video_path}")
    
    # 直接使用已经渲染的帧
    print("Using rendered frames for panorama generation...")
    # 转换frames为numpy格式用于全景图生成
    frames = [(frame.cpu().numpy() * 255).astype(np.uint8) for frame in render_results]
    
    # 转换为全景图
    print(f"Converting frames to panorama using advanced spherical projection...")
    # 传递真实的相机轨迹对象
    panorama, pano_mask = frames_to_panorama(frames, trajectory)
    
    # 保存结果
    pano_path = os.path.join(output_dir, 'panorama.png')
    mask_path = os.path.join(output_dir, 'panorama_mask.png')
    
    Image.fromarray(panorama).save(pano_path)
    Image.fromarray(pano_mask).save(mask_path)
    
    print(f"Panorama saved to {pano_path}")
    print(f"Mask saved to {mask_path}")
    
    return panorama, pano_mask

f = 383.13     #设置焦距，对应67.5°视场角
Mcam.set_default_f(f)
plan = CamPlanner() # 控制相机运镜
pcd=PcdMgr(ply_file_path=f'/home/liujiajun/HunyuanWorld-1.0/test_results/street/pointcloud/panorama_pointcloud.ply')

pcd.pts[:,:3]=rotate_point_cloud(pcd.pts[:,:3], angle_x_deg=90, angle_y_deg=-90, angle_z_deg=0)

# pcd=image2pcd()
PcdMgr.set_default_render_backend('gs')
# plan里面有很多轨迹
# traj2=plan.add_traj().move_orbit_to(0, 360, 0.1, num_frames=96).finish()

# 创建360度环绕轨迹用于全景图生成
traj_orbit = plan.add_traj().move_orbit_to(0, 360, 0.5, num_frames=72).finish()

# 或者使用更复杂的轨迹
# traj_orbit=plan.add_traj().move_forward(0.5, num_frames=96).finish()
# traj_orbit=plan.add_traj(startcam=traj_orbit[-1]).move_orbit_to(0, 360, 0.0001, num_frames=96).finish()

for i in range(len(traj_orbit)):
    traj_orbit[i].set_size(512,512)

# 创建输出目录
os.makedirs('testOutput', exist_ok=True)

# 原始视频渲染 (先获取帧，然后使用easy_save_video保存)
render_results=CamPlanner._render_video(pcd, traj = traj_orbit, output_path=None)
render_mask_results=CamPlanner._render_video(pcd, traj = traj_orbit, output_path=None, mask=True)

# 保存视频
frames_np = torch.stack(render_results).cpu().numpy()  # [T, H, W, C]

# 处理mask - mask是2D的[H, W]，需要扩展到3通道
mask_list = []
for m in render_mask_results:
    if len(m.shape) == 2:  # [H, W]
        # 扩展到3通道
        m_3ch = m.unsqueeze(-1).expand(-1, -1, 3)  # [H, W, 3]
    elif len(m.shape) == 3:
        if m.shape[-1] == 1:  # [H, W, 1]
            m_3ch = m.expand(-1, -1, 3)  # [H, W, 3]
        else:  # already [H, W, 3]
            m_3ch = m
    else:
        print(f"Warning: unexpected mask shape {m.shape}")
        m_3ch = m
    mask_list.append(m_3ch)
    
mask_frames_np = torch.stack(mask_list).cpu().numpy()  # [T, H, W, 3]

easy_save_video(frames_np, 'testOutput/test_video.mp4', fps=30, value_range="0,1")
easy_save_video(mask_frames_np, 'testOutput/test_video_mask.mp4', fps=30, value_range="0,1")

print("Videos saved successfully!")

# 生成全景图
print("\n=== Generating Panorama from Rendered Video ===")

panorama, pano_mask = render_panorama_from_trajectory(pcd, traj_orbit, output_dir='testOutput/panorama_output')

print(f"\n✅ Panorama generated successfully")