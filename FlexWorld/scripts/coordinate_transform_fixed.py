#!/usr/bin/env python3
"""
修复版坐标系变换脚本 - 保持法向量
"""

import sys
import os
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

def save_pointcloud_with_normals(points, colors, normals, output_path):
    """保存带法向量的点云文件"""
    # 构造PLY文件头
    header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property double x
property double y
property double z
property double nx
property double ny  
property double nz
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with open(output_path, 'wb') as f:
        # 写入头部
        f.write(header.encode('ascii'))
        
        # 写入数据 (xyz + nxnynz + rgb)
        for i in range(len(points)):
            # xyz (double)
            f.write(points[i][0].astype(np.float64).tobytes())
            f.write(points[i][1].astype(np.float64).tobytes())
            f.write(points[i][2].astype(np.float64).tobytes())
            
            # nxnynz (double)
            f.write(normals[i][0].astype(np.float64).tobytes())
            f.write(normals[i][1].astype(np.float64).tobytes())
            f.write(normals[i][2].astype(np.float64).tobytes())
            
            # rgb (uchar)
            rgb = (colors[i] * 255).astype(np.uint8)
            f.write(rgb[0].tobytes())
            f.write(rgb[1].tobytes()) 
            f.write(rgb[2].tobytes())

def main():
    parser = argparse.ArgumentParser(description='Apply coordinate transformation to pointcloud (FIXED VERSION)')
    parser.add_argument('--input', required=True, help='Input pointcloud path')
    parser.add_argument('--output', required=True, help='Output pointcloud path')
    parser.add_argument('--angle_x', type=float, default=90, help='X rotation angle in degrees')
    parser.add_argument('--angle_y', type=float, default=-90, help='Y rotation angle in degrees')
    parser.add_argument('--angle_z', type=float, default=0, help='Z rotation angle in degrees')
    
    args = parser.parse_args()

    print("🔧 使用修复版坐标变换（保持法向量）")

    # 读取原始点云文件（带法向量）
    try:
        pcd_original = o3d.io.read_point_cloud(args.input)
        
        # 检查是否有法向量
        has_normals = pcd_original.has_normals()
        print(f"📊 原始文件: {len(pcd_original.points)} 点, 法向量: {'是' if has_normals else '否'}")
        
        if not has_normals:
            print("⚠️  原始文件没有法向量，估算法向量...")
            pcd_original.estimate_normals()
        
        points = np.asarray(pcd_original.points)
        colors = np.asarray(pcd_original.colors)
        normals = np.asarray(pcd_original.normals)
        
        # 应用坐标变换（同时变换点和法向量）
        print(f"🔄 应用旋转: X={args.angle_x}°, Y={args.angle_y}°, Z={args.angle_z}°")
        
        transformed_points = rotate_point_cloud(points, args.angle_x, args.angle_y, args.angle_z)
        transformed_normals = rotate_point_cloud(normals, args.angle_x, args.angle_y, args.angle_z)
        
        # 保存带法向量的点云
        save_pointcloud_with_normals(transformed_points, colors, transformed_normals, args.output)
        
        # 验证保存结果
        saved_size = os.path.getsize(args.output) / (1024*1024)
        original_size = os.path.getsize(args.input) / (1024*1024)
        
        print(f"✅ 保存完成:")
        print(f"   点数: {len(transformed_points)}")
        print(f"   原始大小: {original_size:.2f} MB")
        print(f"   保存大小: {saved_size:.2f} MB")
        print(f"   大小变化: {saved_size - original_size:+.2f} MB")
        
        if abs(saved_size - original_size) < 1.0:
            print("✅ 文件大小保持一致，法向量保留成功！")
        else:
            print("⚠️  文件大小有差异，可能存在精度损失")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()
