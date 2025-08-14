#!/usr/bin/env python3
"""
点云更新脚本
从incremental_pipeline.py中提取的update_pointcloud功能
"""

import sys
import argparse
import numpy as np
import cv2
import open3d as o3d
import os
import json

sys.path.append('/home/liujiajun/HunyuanWorld-1.0/FlexWorld')
from ops.PcdMgr import PcdMgr
from ops.cam_utils import Mcam
from ops.utils.depth import refine_depth2, depth2pcd_world

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

def main():
    parser = argparse.ArgumentParser(description='Update pointcloud with inpainted pixels')
    parser.add_argument('--current_pointcloud', required=True, help='Current pointcloud path')
    parser.add_argument('--inpainted_rgb', required=True, help='Inpainted RGB image path')
    parser.add_argument('--original_depth', required=True, help='Original depth numpy file path')
    parser.add_argument('--mask_path', required=True, help='Mask image path')
    parser.add_argument('--output_pointcloud', required=True, help='Output pointcloud path')
    parser.add_argument('--frame_idx', type=int, required=True, help='Frame index')
    parser.add_argument('--cam_R', required=True, help='Camera rotation matrix (JSON string)')
    parser.add_argument('--cam_T', required=True, help='Camera translation vector (JSON string)')
    parser.add_argument('--cam_f', type=float, required=True, help='Camera focal length')
    parser.add_argument('--cam_c', required=True, help='Camera principal point (JSON string)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    if args.debug:
        print(f"\n🔍 ===== FRAME {args.frame_idx:03d} DEBUG INFO =====")

    # 解析相机参数
    cam_R = np.array(json.loads(args.cam_R))
    cam_T = np.array(json.loads(args.cam_T))
    cam_c = np.array(json.loads(args.cam_c))

    # 加载当前点云
    current_pcd = PcdMgr(ply_file_path=args.current_pointcloud)
    original_point_count = len(current_pcd.pts)
    pointcloud_size_mb = os.path.getsize(args.current_pointcloud) / (1024 * 1024)
    if args.debug:
        print(f"📊 Original pointcloud: {original_point_count} points, {pointcloud_size_mb:.1f} MB")

    # 加载数据
    inpainted_rgb = cv2.imread(args.inpainted_rgb)
    inpainted_rgb = cv2.cvtColor(inpainted_rgb, cv2.COLOR_BGR2RGB)
    original_depth = np.load(args.original_depth)
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE) > 0

    if args.debug:
        print(f"📷 Depth range: [{original_depth.min():.3f}, {original_depth.max():.3f}]")
        print(f"🎭 Mask coverage: {np.sum(mask)} / {mask.size} pixels ({np.sum(mask)/mask.size*100:.2f}%)")

    # 重建相机对象（与渲染时完全相同）
    cam = Mcam()
    cam.R = cam_R
    cam.T = cam_T
    cam.f = args.cam_f
    cam.c = cam_c
    cam.set_size(512, 512)

    if args.debug:
        print(f"📹 Camera: f={cam.f}, c={cam.c}")
        print(f"📍 Camera pos: [{cam.T[0]:.3f}, {cam.T[1]:.3f}, {cam.T[2]:.3f}]")

    points_added = 0
    
    if np.sum(mask) > 0:
        # 估计深度（简单插值方法）
        estimated_depth = cv2.inpaint(
            original_depth.astype(np.float32),
            mask.astype(np.uint8) * 255,
            inpaintRadius=10,
            flags=cv2.INPAINT_TELEA
        )
        
        if args.debug:
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
        
        if args.debug:
            print(f"⚖️  Aligned depth range: [{aligned_depth.min():.3f}, {aligned_depth.max():.3f}]")
        
        # 3D重建
        points_3d = depth2pcd_world(aligned_depth, cam)
        
        # 只添加mask区域的点
        new_points_3d = points_3d[mask]
        new_colors = inpainted_rgb[mask] / 255.0
        new_points_6d = np.concatenate([new_points_3d, new_colors], axis=-1)
        
        if args.debug:
            print(f"🎯 New points generated: {len(new_points_6d)}")
        
        if len(new_points_6d) > 0:
            if args.debug:
                # 分析新增点的空间分布
                x_range = [new_points_3d[:, 0].min(), new_points_3d[:, 0].max()]
                y_range = [new_points_3d[:, 1].min(), new_points_3d[:, 1].max()]
                z_range = [new_points_3d[:, 2].min(), new_points_3d[:, 2].max()]
                
                print(f"📐 New points X: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
                print(f"📐 New points Y: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
                print(f"📐 New points Z: [{z_range[0]:.3f}, {z_range[1]:.3f}]")
                
                print("⚠️  SKIPPING outlier removal for debugging!")
            
            current_pcd.add_pts(new_points_6d)
            points_added = len(new_points_6d)
            
            if args.debug:
                print(f"✅ Added {points_added} points directly (no filtering)")
        else:
            if args.debug:
                print("❌ No points to add")
    else:
        if args.debug:
            print("⏭️  No missing pixels, skipping depth estimation")

    # 保存更新后的点云
    pts = current_pcd.pts
    final_point_count = len(pts)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(pts[:, 3:6])
    o3d.io.write_point_cloud(args.output_pointcloud, pcd_o3d)

    # 计算文件大小
    final_size_mb = os.path.getsize(args.output_pointcloud) / (1024 * 1024)
    size_change = final_size_mb - pointcloud_size_mb

    if args.debug:
        print(f"💾 Final pointcloud: {final_point_count} points, {final_size_mb:.1f} MB")
        print(f"📈 Change: +{final_point_count - original_point_count} points, {size_change:+.1f} MB")
        print(f"🎯 Points added this frame: {points_added}")
        print(f"===== FRAME {args.frame_idx:03d} COMPLETED =====\n")
    else:
        print(f"Points: {original_point_count} → {final_point_count} (+{points_added}), "
              f"Size: {pointcloud_size_mb:.1f}MB → {final_size_mb:.1f}MB ({size_change:+.1f}MB)")

if __name__ == "__main__":
    main()