#!/usr/bin/env python3
"""
坐标系变换脚本
从incremental_pipeline.py中提取的_apply_coordinate_transform功能
"""

import sys
import argparse
import numpy as np
import open3d as o3d

sys.path.append('/home/liujiajun/HunyuanWorld-1.0/FlexWorld')
from ops.PcdMgr import PcdMgr

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
    parser = argparse.ArgumentParser(description='Apply coordinate transformation to pointcloud')
    parser.add_argument('--input', required=True, help='Input pointcloud path')
    parser.add_argument('--output', required=True, help='Output pointcloud path')
    parser.add_argument('--angle_x', type=float, default=90, help='X rotation angle in degrees')
    parser.add_argument('--angle_y', type=float, default=-90, help='Y rotation angle in degrees')
    parser.add_argument('--angle_z', type=float, default=0, help='Z rotation angle in degrees')
    
    args = parser.parse_args()

    # 加载点云
    pcd = PcdMgr(ply_file_path=args.input)
    original_count = len(pcd.pts)
    print(f"Loaded pointcloud: {original_count} points")

    # 应用坐标系变换（对齐ljj.py）
    pcd.pts[:,:3] = rotate_point_cloud(pcd.pts[:,:3], 
                                      angle_x_deg=args.angle_x, 
                                      angle_y_deg=args.angle_y, 
                                      angle_z_deg=args.angle_z)

    # 保存变换后的点云
    pts = pcd.pts
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(pts[:, 3:6])
    o3d.io.write_point_cloud(args.output, pcd_o3d)

    print(f"Coordinate transformation applied: {original_count} → {len(pts)} points")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()