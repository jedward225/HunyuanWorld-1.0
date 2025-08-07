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


def frames_to_panorama_center_align(frames, camera_params, output_width=1920, output_height=960):
    """
    将渲染的视频帧转换为全景图及其mask - 中心对齐方式。
    帧内容保持原始大小，垂直居中放置在全景图中。
    
    参数:
    frames: 视频帧列表或numpy数组 [N, H, W, C]
    camera_params: 相机参数列表（每帧对应的相机位置/朝向）
    output_width: 输出全景图宽度
    output_height: 输出全景图高度
    
    返回:
    panorama: 全景图 numpy数组
    mask: 有效像素mask
    """
    if len(frames) == 0:
        return np.zeros((output_height, output_width, 3), dtype=np.uint8), np.zeros((output_height, output_width), dtype=np.uint8)
    
    # 获取输入帧的尺寸
    frame_h, frame_w = frames[0].shape[:2]
    
    # 初始化全景图和权重图
    panorama = np.zeros((output_height, output_width, 3), dtype=np.float32)
    weights = np.zeros((output_height, output_width), dtype=np.float32)
    
    # 计算FOV（视场角）
    fov_horizontal = np.deg2rad(67.5)  # 水平FOV，可根据相机参数调整
    
    # 假设相机做360度旋转
    num_frames = len(frames)
    
    # 计算垂直居中的起始位置
    y_offset = (output_height - frame_h) // 2
    y_start = max(0, y_offset)
    y_end = min(output_height, y_offset + frame_h)
    
    # 如果帧高度大于全景图，需要裁剪
    frame_y_start = 0 if y_offset >= 0 else -y_offset
    frame_y_end = frame_y_start + (y_end - y_start)
    
    for i, frame in enumerate(frames):
        # 计算当前帧对应的方位角
        azimuth = (i / num_frames) * 2 * np.pi  # 0 to 2π
        
        # 计算此帧在全景图中的中心列位置
        center_col = int((azimuth / (2 * np.pi)) * output_width)
        
        # 计算此帧在全景图中占据的宽度
        pano_width_per_frame = int(output_width * fov_horizontal / (2 * np.pi))
        
        # 计算在全景图中的起始和结束列
        start_col = center_col - pano_width_per_frame // 2
        end_col = start_col + pano_width_per_frame
        
        # 将帧内容投影到全景图（中心对齐）
        for j in range(start_col, end_col):
            col_idx = j % output_width  # 处理环绕
            
            # 计算对应的源图像列
            src_col = int((j - start_col) * frame_w / pano_width_per_frame)
            if 0 <= src_col < frame_w:
                # 复制列数据，垂直居中
                panorama[y_start:y_end, col_idx] += frame[frame_y_start:frame_y_end, src_col]
                weights[y_start:y_end, col_idx] += 1.0
    
    # 归一化（平均重叠区域）
    mask = weights > 0
    panorama[mask] = panorama[mask] / weights[mask, np.newaxis]
    
    # 转换为uint8
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)
    mask = mask.astype(np.uint8) * 255
    
    return panorama, mask


def frames_to_panorama_scaled(frames, camera_params, output_width=1920, output_height=960):
    """
    将渲染的视频帧转换为全景图及其mask - 缩放方式。
    帧内容被缩放以匹配全景图高度。
    
    参数:
    frames: 视频帧列表或numpy数组 [N, H, W, C]
    camera_params: 相机参数列表（每帧对应的相机位置/朝向）
    output_width: 输出全景图宽度
    output_height: 输出全景图高度
    
    返回:
    panorama: 全景图 numpy数组
    mask: 有效像素mask
    """
    if len(frames) == 0:
        return np.zeros((output_height, output_width, 3), dtype=np.uint8), np.zeros((output_height, output_width), dtype=np.uint8)
    
    # 获取输入帧的尺寸
    frame_h, frame_w = frames[0].shape[:2]
    
    # 如果帧高度与全景图高度不同，先缩放所有帧
    if frame_h != output_height:
        scale_factor = output_height / frame_h
        new_width = int(frame_w * scale_factor)
        scaled_frames = []
        for frame in frames:
            # 使用双线性插值缩放
            scaled_frame = cv2.resize(frame, (new_width, output_height), interpolation=cv2.INTER_LINEAR)
            scaled_frames.append(scaled_frame)
        frames = scaled_frames
        frame_w = new_width
        frame_h = output_height
    
    # 初始化全景图和权重图
    panorama = np.zeros((output_height, output_width, 3), dtype=np.float32)
    weights = np.zeros((output_height, output_width), dtype=np.float32)
    
    # 计算FOV（视场角）
    fov_horizontal = np.deg2rad(67.5)  # 水平FOV，可根据相机参数调整
    
    # 假设相机做360度旋转
    num_frames = len(frames)
    
    for i, frame in enumerate(frames):
        # 计算当前帧对应的方位角
        azimuth = (i / num_frames) * 2 * np.pi  # 0 to 2π
        
        # 计算此帧在全景图中的中心列位置
        center_col = int((azimuth / (2 * np.pi)) * output_width)
        
        # 计算此帧在全景图中占据的宽度
        pano_width_per_frame = int(output_width * fov_horizontal / (2 * np.pi))
        
        # 计算在全景图中的起始和结束列
        start_col = center_col - pano_width_per_frame // 2
        end_col = start_col + pano_width_per_frame
        
        # 将帧内容投影到全景图
        for j in range(start_col, end_col):
            col_idx = j % output_width  # 处理环绕
            
            # 计算对应的源图像列
            src_col = int((j - start_col) * frame_w / pano_width_per_frame)
            if 0 <= src_col < frame_w:
                # 复制整列数据（已缩放到正确高度）
                panorama[:, col_idx] += frame[:, src_col]
                weights[:, col_idx] += 1.0
    
    # 归一化（平均重叠区域）
    mask = weights > 0
    panorama[mask] = panorama[mask] / weights[mask, np.newaxis]
    
    # 转换为uint8
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)
    mask = mask.astype(np.uint8) * 255
    
    return panorama, mask


# 保留原函数名作为默认选择
def frames_to_panorama(frames, camera_params, output_width=1920, output_height=960, mode="center"):
    """
    将渲染的视频帧转换为全景图及其mask。
    
    参数:
    frames: 视频帧列表或numpy数组 [N, H, W, C]
    camera_params: 相机参数列表（每帧对应的相机位置/朝向）
    output_width: 输出全景图宽度
    output_height: 输出全景图高度
    mode: "center" for center-align, "scale" for scaling
    
    返回:
    panorama: 全景图 numpy数组
    mask: 有效像素mask
    """
    if mode == "scale":
        return frames_to_panorama_scaled(frames, camera_params, output_width, output_height)
    else:
        return frames_to_panorama_center_align(frames, camera_params, output_width, output_height)


def extract_frames_from_video(video_path, skip_frames=1):
    """
    从视频文件中提取帧。
    
    参数:
    video_path: 视频文件路径
    skip_frames: 跳帧数（1表示提取所有帧）
    
    返回:
    frames: 帧列表
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip_frames == 0:
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    return frames


def render_panorama_from_trajectory(pcd, trajectory, output_dir='testOutput/panorama_output', pano_mode='center'):
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
    mask_frames = [(frame.cpu().numpy() * 255).astype(np.uint8) for frame in render_mask]
    
    # 转换为全景图
    print(f"Converting frames to panorama using {pano_mode} mode...")
    # 传递真实的相机轨迹对象
    panorama, pano_mask = frames_to_panorama(frames, trajectory, mode=pano_mode)
    
    # 保存结果
    pano_path = os.path.join(output_dir, 'panorama.png')
    mask_path = os.path.join(output_dir, 'panorama_mask.png')
    
    Image.fromarray(panorama).save(pano_path)
    Image.fromarray(pano_mask).save(mask_path)
    
    print(f"Panorama saved to {pano_path}")
    print(f"Mask saved to {mask_path}")
    
    return panorama, pano_mask


f = 200     #设置焦距，随便设，探索有啥影响
Mcam.set_default_f(f)
plan = CamPlanner() # 控制相机运镜
pcd=PcdMgr(ply_file_path=f'/home/liujiajun/HunyuanWorld-1.0/test_results/livingroom/pointcloud/panorama_pointcloud.ply')

pcd.pts[:,:3]=rotate_point_cloud(pcd.pts[:,:3], angle_x_deg=90, angle_y_deg=-90, angle_z_deg=0)

# pcd=image2pcd()
PcdMgr.set_default_render_backend('gs')
# plan里面有很多轨迹
# traj2=plan.add_traj().move_orbit_to(0, 360, 0.1, num_frames=96).finish()

# 创建360度环绕轨迹用于全景图生成
traj_orbit = plan.add_traj().move_orbit_to(0, 360, 0.1, num_frames=96).finish()

# 或者使用更复杂的轨迹
# traj2=plan.add_traj().move_forward(0.5, num_frames=96).finish()
# traj2=plan.add_traj(startcam=traj2[-1]).move_orbit_to(0, 360, 0.0001, num_frames=96).finish()

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

# 选择模式: "center" for center-align, "scale" for scaling
PANO_MODE = "center"  # 或 "scale"

panorama, pano_mask = render_panorama_from_trajectory(pcd, traj_orbit, output_dir='testOutput/panorama_output', pano_mode=PANO_MODE)

print(f"\nPanorama generated using {PANO_MODE} mode.")
print("To change mode, edit PANO_MODE variable to 'center' or 'scale'")
