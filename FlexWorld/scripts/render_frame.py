#!/usr/bin/env python3
"""
单帧渲染脚本
从incremental_pipeline.py中提取的render_frame功能
"""

import sys
import argparse
import numpy as np
import cv2
import json
import einops

sys.path.append('/home/liujiajun/HunyuanWorld-1.0/FlexWorld')
from ops.PcdMgr import PcdMgr
from ops.cam_utils import Mcam

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
    parser = argparse.ArgumentParser(description='Render frame from pointcloud')
    parser.add_argument('--pointcloud', required=True, help='Input pointcloud path')
    parser.add_argument('--output_dir', required=True, help='Output directory for frames')
    parser.add_argument('--frame_idx', type=int, required=True, help='Frame index')
    parser.add_argument('--cam_R', required=True, help='Camera rotation matrix (JSON string)')
    parser.add_argument('--cam_T', required=True, help='Camera translation vector (JSON string)')
    parser.add_argument('--cam_f', type=float, required=True, help='Camera focal length')
    parser.add_argument('--cam_c', required=True, help='Camera principal point (JSON string)')
    
    args = parser.parse_args()

    # 解析相机参数
    cam_R = np.array(json.loads(args.cam_R))
    cam_T = np.array(json.loads(args.cam_T))
    cam_c = np.array(json.loads(args.cam_c))

    # 加载点云（坐标系变换只在初始化时应用）
    pcd = PcdMgr(ply_file_path=args.pointcloud)

    # 重建相机对象（使用传入的相机数据）
    cam = Mcam()
    cam.R = cam_R
    cam.T = cam_T
    cam.f = args.cam_f
    cam.c = cam_c
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
    cv2.imwrite(f"{args.output_dir}/frame_{args.frame_idx:03d}.png", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{args.output_dir}/alpha_{args.frame_idx:03d}.png", alpha_img)
    np.save(f"{args.output_dir}/depth_{args.frame_idx:03d}.npy", depth_img)

    # 生成mask（alpha<10的区域需要inpaint）
    mask = alpha_img < 10
    cv2.imwrite(f"{args.output_dir}/mask_{args.frame_idx:03d}.png", mask.astype(np.uint8) * 255)

    # 输出统计
    stats = {
        "missing_pixels": int(np.sum(mask)),
        "total_pixels": 512 * 512,
        "missing_ratio": float(np.sum(mask) / (512 * 512))
    }

    print(json.dumps(stats))

if __name__ == "__main__":
    main()