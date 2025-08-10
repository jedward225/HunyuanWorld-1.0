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
from ops.utils.pano_tool import multi_Perspec2Equirec as m_P2E
import tempfile

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

def video_to_panorama_flexworld(video_frames, output_dir, fov=90):
    """
    使用FlexWorld的video2pano方法直接从视频帧生成全景图
    
    参数:
    video_frames: 视频帧列表 (torch tensors)
    output_dir: 输出目录
    fov: 视场角
    """
    from ops.utils.pano import video2pano
    import tempfile
    import shutil
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建临时目录保存视频帧
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    # 从360度视频中选择8个关键帧（每45度一帧）
    num_frames = len(video_frames)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    for i, angle in enumerate(angles):
        frame_idx = int((angle / 360.0) * num_frames) % num_frames
        frame = video_frames[frame_idx]
        
        # 转换为numpy并保存
        frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
        img_path = os.path.join(temp_dir, f"frame_{angle:03d}.png")
        cv2.imwrite(img_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
        image_paths.append(img_path)
    
    print(f"Using FlexWorld's video2pano with {len(image_paths)} frames")
    
    # 使用FlexWorld的原生video2pano方法
    pano = video2pano(image_paths, output_dir, fov=fov)
    
    # 清理临时文件
    shutil.rmtree(temp_dir)
    
    # 保存最终全景图
    pano_path = os.path.join(output_dir, 'panorama.png')
    
    print(f"Panorama saved to {pano_path}")
    
    return pano

# 设置相机参数
frame_size = 512  # 帧尺寸
fov = 67.5  # 视场角（度）
# f = 383.13     #设置焦距，对应67.5°视场角
f = frame_size / (2 * np.tan(np.radians(fov/2))) # 从FOV计算焦距: f = (w/2) / tan(fov/2)
print(f"Camera settings: FOV={fov}°, focal_length={f:.2f}, frame_size={frame_size}x{frame_size}")

Mcam.set_default_f(f)
plan = CamPlanner() # 控制相机运镜
# pcd=PcdMgr(ply_file_path=f'/home/liujiajun/HunyuanWorld-1.0/test_results/street/pointcloud/panorama_pointcloud.ply')
pcd=PcdMgr(ply_file_path=f'/home/liujiajun/HunyuanWorld-1.0/test_results/street/pointcloud/panorama_pointcloud.ply')

pcd.pts[:,:3]=rotate_point_cloud(pcd.pts[:,:3], angle_x_deg=90, angle_y_deg=-90, angle_z_deg=0)

# pcd=image2pcd()
PcdMgr.set_default_render_backend('gs')
# plan里面有很多轨迹
# traj2=plan.add_traj().move_orbit_to(0, 360, 0.1, num_frames=96).finish()

# 创建360度环绕轨迹用于全景图生成
# traj_orbit = plan.add_traj().move_orbit_to(0, 360, 0.5, num_frames=72).finish()
traj_orbit = plan.add_traj().move_orbit_to(0, -360, 0.5, num_frames=72).finish()

# 或者使用更复杂的轨迹
# traj_orbit=plan.add_traj().move_forward(0.5, num_frames=96).finish()
# traj_orbit=plan.add_traj(startcam=traj_orbit[-1]).move_orbit_to(0, 360, 0.0001, num_frames=96).finish()

for i in range(len(traj_orbit)):
    traj_orbit[i].set_size(frame_size, frame_size)

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

# 逐帧保存功能 - 保存每一帧和对应的mask
print("\n=== Saving individual frames and masks ===")
frames_output_dir = 'testOutput/frames'
os.makedirs(frames_output_dir, exist_ok=True)

print(f"Saving {len(render_results)} frames...")
for i, (frame, mask) in enumerate(zip(render_results, mask_frames_np)):
    # 保存RGB帧
    frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
    frame_path = os.path.join(frames_output_dir, f'frame_{i:03d}.png')
    cv2.imwrite(frame_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
    
    # 保存mask帧
    mask_np = (mask[..., 0] * 255).astype(np.uint8)  # 使用第一个通道
    mask_path = os.path.join(frames_output_dir, f'mask_{i:03d}.png')
    cv2.imwrite(mask_path, mask_np)
    
    # 保存反向mask（黑色=需要inpainting）
    inverted_mask_np = 255 - mask_np
    inverted_mask_path = os.path.join(frames_output_dir, f'inpaint_mask_{i:03d}.png')
    cv2.imwrite(inverted_mask_path, inverted_mask_np)
    
    if (i + 1) % 10 == 0:
        print(f"  Saved {i + 1}/{len(render_results)} frames")

print(f"✅ All frames saved to: {frames_output_dir}/")
print(f"  - frame_XXX.png: RGB frames")
print(f"  - mask_XXX.png: Coverage masks (white=valid, black=missing)")  
print(f"  - inpaint_mask_XXX.png: Inpainting masks (black=fill, white=keep)")

# 生成全景图 - 从已有视频帧中选择8帧
print("\n=== Generating Panorama from Video Frames ===")

from ops.utils.pano import video2pano
import tempfile
import shutil

# 从72帧轨道视频中选择8帧（每45度一帧）
angles = [0, 45, 90, 135, 180, 225, 270, 315]
temp_dir = tempfile.mkdtemp()
image_paths = []

print("Selecting frames from orbit video...")
for angle in angles:
    # 计算对应的帧索引
    frame_idx = int((angle / 360.0) * len(render_results)) % len(render_results)
    frame = render_results[frame_idx]
    
    # 保存帧 - 确保正确的颜色转换
    frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
    
    # 调试：检查帧是否是彩色的
    print(f"  Frame {frame_idx} shape: {frame_np.shape}, is_color: {frame_np.shape[-1] == 3 if len(frame_np.shape) == 3 else False}")
    
    img_path = os.path.join(temp_dir, f"frame_{angle:03d}.png")
    # 确保正确的颜色格式
    if len(frame_np.shape) == 3 and frame_np.shape[-1] == 3:
        cv2.imwrite(img_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
    else:
        print(f"  Warning: Frame {frame_idx} is not RGB format!")
        cv2.imwrite(img_path, frame_np)
    
    image_paths.append(img_path)
    print(f"  Frame {frame_idx} -> {angle}° view")

# 调用FlexWorld的video2pano
pano_output_dir = 'testOutput/panorama_output'
os.makedirs(pano_output_dir, exist_ok=True)
panorama = video2pano(image_paths, pano_output_dir, fov=fov)

# 生成mask版本的全景图
print("Generating mask panorama for inpainting...")

# 从mask_frames_np中选择对应的8帧
mask_temp_dir = tempfile.mkdtemp()
mask_image_paths = []

for angle in angles:
    frame_idx = int((angle / 360.0) * len(mask_frames_np)) % len(mask_frames_np)
    mask_frame = mask_frames_np[frame_idx]
    
    # 转换为0-255范围的单通道mask (白色=有点云，黑色=无点云)
    mask_gray = (mask_frame[..., 0] * 255).astype(np.uint8)
    mask_3ch = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    
    # 保存mask帧
    mask_img_path = os.path.join(mask_temp_dir, f"mask_frame_{angle:03d}.png")
    cv2.imwrite(mask_img_path, mask_3ch)
    mask_image_paths.append(mask_img_path)
    print(f"  Mask frame {frame_idx} -> {angle}° view")

# 使用video2pano生成mask全景图 (使用临时目录，避免覆盖彩色pano.png)
mask_temp_output_dir = tempfile.mkdtemp()
mask_panorama_full = video2pano(mask_image_paths, mask_temp_output_dir, fov=fov)

# 裁剪并调整mask尺寸，与pano.png保持一致 (1280x576)
mask_panorama_cropped = mask_panorama_full[650:-650]  # 裁剪高度
mask_panorama_resized = cv2.resize(mask_panorama_cropped, (1280, 576), interpolation=cv2.INTER_LINEAR)

# 保存mask全景图 (与pano.png同样尺寸)
mask_pano_path = os.path.join(pano_output_dir, 'mask_for_inpainting.png')
cv2.imwrite(mask_pano_path, mask_panorama_resized.astype(np.uint8))

# 生成反向mask (黑色=需要inpainting的区域)
inverted_mask = 255 - mask_panorama_resized
inverted_mask_path = os.path.join(pano_output_dir, 'inpaint_mask.png')
cv2.imwrite(inverted_mask_path, inverted_mask.astype(np.uint8))

# 用于统计的mask
mask_panorama = mask_panorama_resized

# 清理临时文件
shutil.rmtree(temp_dir)
shutil.rmtree(mask_temp_dir)
shutil.rmtree(mask_temp_output_dir)

print(f"\n✅ Panorama generated successfully!")
print(f"  - Original panorama: {pano_output_dir}/pano.png")
print(f"  - Coverage mask: {mask_pano_path}")
print(f"  - Inpainting mask: {inverted_mask_path}")

# 计算覆盖率统计
mask_gray = cv2.cvtColor(mask_panorama.astype(np.uint8), cv2.COLOR_BGR2GRAY)
total_pixels = mask_gray.shape[0] * mask_gray.shape[1]
valid_pixels = np.sum(mask_gray > 128)
coverage_ratio = valid_pixels / total_pixels * 100
print(f"  - Coverage: {coverage_ratio:.1f}% ({valid_pixels}/{total_pixels} pixels)")