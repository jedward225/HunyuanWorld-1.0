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


f = 200     #设置焦距，随便设，探索有啥影响
Mcam.set_default_f(f)
plan = CamPlanner() # 控制相机运镜
pcd=PcdMgr(ply_file_path=f'/mnt/zhouzihan/BigOne/cache/panorama_pointcloud.ply')

pcd.pts[:,:3]=rotate_point_cloud(pcd.pts[:,:3], angle_x_deg=90, angle_y_deg=-90, angle_z_deg=0)

# pcd=image2pcd()
PcdMgr.set_default_render_backend('gs')
# plan里面有很多轨迹
# traj2=plan.add_traj().move_orbit_to(0, 360, 0.1, num_frames=96).finish()

traj2=plan.add_traj().move_forward(0.5, num_frames=96).finish()
traj2=plan.add_traj(startcam=traj2[-1]).move_orbit_to(0, 360, 0.0001, num_frames=96).finish()

for i in range(len(traj2)):
    traj2[i].set_size(512,512)

render_results=CamPlanner._render_video(pcd, traj = traj2,output_path='cache/777.mp4')
render_results=CamPlanner._render_video(pcd, traj = traj2,output_path='cache/777_mask.mp4',mask=True)
