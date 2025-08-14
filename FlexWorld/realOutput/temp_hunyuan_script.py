
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

print(f"\n🔍 ===== FRAME 001 DEBUG INFO =====")

# 加载当前点云
current_pcd = PcdMgr(ply_file_path="/home/liujiajun/HunyuanWorld-1.0/FlexWorld/realOutput/pointclouds/pointcloud_current.ply")
original_point_count = len(current_pcd.pts)
pointcloud_size_mb = os.path.getsize("/home/liujiajun/HunyuanWorld-1.0/FlexWorld/realOutput/pointclouds/pointcloud_current.ply") / (1024 * 1024)
print(f"📊 Original pointcloud: {original_point_count} points, {pointcloud_size_mb:.1f} MB")

# 加载数据
inpainted_rgb = cv2.imread("/home/liujiajun/HunyuanWorld-1.0/FlexWorld/realOutput/inpainted/inpainted_001.png")
inpainted_rgb = cv2.cvtColor(inpainted_rgb, cv2.COLOR_BGR2RGB)
original_depth = np.load("/home/liujiajun/HunyuanWorld-1.0/FlexWorld/realOutput/frames/depth_001.npy")
mask = cv2.imread("/home/liujiajun/HunyuanWorld-1.0/FlexWorld/realOutput/frames/mask_001.png", cv2.IMREAD_GRAYSCALE) > 0

print(f"📷 Depth range: [{original_depth.min():.3f}, {original_depth.max():.3f}]")
print(f"🎭 Mask coverage: {np.sum(mask)} / {mask.size} pixels ({np.sum(mask)/mask.size*100:.2f}%)")

# 重建相机对象（与渲染时完全相同）
from ops.cam_utils import Mcam
cam = Mcam()
cam.R = np.array([[0.9961947202682495, 0.0, -0.08715574443340302], [-0.0, 1.0, 0.0], [0.08715574443340302, 0.0, 0.9961947202682495]])
cam.T = np.array([-0.04357787221670151, 0.0, -0.0019026509253308177])
cam.f = 383.1310752423652
cam.c = np.array([256, 256])
cam.set_size(512, 512)

print(f"📹 Camera: f={cam.f}, c={cam.c}")
print(f"📍 Camera pos: [{cam.T[0]:.3f}, {cam.T[1]:.3f}, {cam.T[2]:.3f}]")

if np.sum(mask) > 0:
    # 估计深度（简单插值方法）
    estimated_depth = cv2.inpaint(
        original_depth.astype(np.float32),
        mask.astype(np.uint8) * 255,
        inpaintRadius=10,
        flags=cv2.INPAINT_TELEA
    )
    
    print(f"🔧 Estimated depth range: [{estimated_depth.min():.3f}, {estimated_depth.max():.3f}]")
    
    # 深度对齐
    aligned_depth = refine_depth2(
        render_dpt=original_depth,
        ipaint_dpt=estimated_depth,
        ipaint_msk=mask,
        iters=50,
        blur_size=15,
        scaled=True
    )
    
    print(f"⚖️  Aligned depth range: [{aligned_depth.min():.3f}, {aligned_depth.max():.3f}]")
    
    # 3D重建
    points_3d = depth2pcd_world(aligned_depth, cam)
    
    # 只添加mask区域的点
    new_points_3d = points_3d[mask]
    new_colors = inpainted_rgb[mask] / 255.0
    new_points_6d = np.concatenate([new_points_3d, new_colors], axis=-1)
    
    print(f"🎯 New points generated: {len(new_points_6d)}")
    
    if len(new_points_6d) > 0:
        # 分析新增点的空间分布
        x_range = [new_points_3d[:, 0].min(), new_points_3d[:, 0].max()]
        y_range = [new_points_3d[:, 1].min(), new_points_3d[:, 1].max()]
        z_range = [new_points_3d[:, 2].min(), new_points_3d[:, 2].max()]
        
        print(f"📐 New points X: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
        print(f"📐 New points Y: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
        print(f"📐 New points Z: [{z_range[0]:.3f}, {z_range[1]:.3f}]")
        
        # ⚠️ 移除离群点过滤，直接添加
        print("⚠️  SKIPPING outlier removal for debugging!")
        current_pcd.add_pts(new_points_6d)
        points_added = len(new_points_6d)
        
        print(f"✅ Added {points_added} points directly (no filtering)")
    else:
        points_added = 0
        print("❌ No points to add")
else:
    points_added = 0
    print("⏭️  No missing pixels, skipping depth estimation")

# 保存更新后的点云（根据保存模式）
if True:
    output_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/realOutput/pointclouds/pointcloud_current.ply"
else:
    output_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/realOutput/pointclouds/pointcloud_002.ply"

pts = current_pcd.pts
final_point_count = len(pts)

# 检查是否有法向量数据（PcdMgr可能包含法向量）
if pts.shape[1] >= 9:  # x,y,z,nx,ny,nz,r,g,b
    print("💾 保存点云（包含法向量）")
    # 使用原始文件格式保存法向量
    # 读取原始文件格式作为模板
    template_pcd = o3d.io.read_point_cloud("/home/liujiajun/HunyuanWorld-1.0/FlexWorld/street_pointcloud_new.ply")
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

print(f"💾 Final pointcloud: {final_point_count} points, {final_size_mb:.1f} MB")
print(f"📈 Change: +{final_point_count - original_point_count} points, {size_change:+.1f} MB")
print(f"🎯 Points added this frame: {points_added}")
print(f"===== FRAME 001 COMPLETED =====\n")
