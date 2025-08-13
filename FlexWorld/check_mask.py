#!/usr/bin/env python3
"""
检测mask中需要inpaint的像素数量
"""

import cv2
import numpy as np

def analyze_mask(mask_path, image_path):
    """分析mask和对应图像"""
    
    # 读取mask和图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    
    print(f"📸 图像尺寸: {image.shape}")
    print(f"🎭 Mask尺寸: {mask.shape}")
    
    # 计算mask统计
    total_pixels = mask.shape[0] * mask.shape[1]
    
    # 基于FlexWorld原理：只有完全透明(值=0)的像素才需要inpaint
    # 值=0: 完全没有点云信息，需要inpaint
    # 值>0: 有点云信息，不需要inpaint
    
    need_inpaint = np.sum(mask == 0)     # 完全透明，需要补全
    has_geometry = np.sum(mask > 0)      # 有点云信息，不需要补全
    
    inpaint_ratio = need_inpaint / total_pixels * 100
    geometry_ratio = has_geometry / total_pixels * 100
    
    print(f"\n📊 基于FlexWorld Alpha的Mask分析:")
    print(f"  总像素数: {total_pixels:,}")
    print(f"  需要补全(=0): {need_inpaint:,} ({inpaint_ratio:.2f}%)")
    print(f"  有几何信息(>0): {has_geometry:,} ({geometry_ratio:.2f}%)")
    
    # 检查mask的值分布
    unique_values = np.unique(mask)
    print(f"  Mask中的唯一值: {unique_values}")
    
    # 可视化mask分布
    print(f"\n🎨 像素值分布:")
    for val in unique_values:
        count = np.sum(mask == val)
        ratio = count / total_pixels * 100
        print(f"    值={val}: {count:,}像素 ({ratio:.2f}%)")
    
    # 保存可视化结果
    mask_vis = np.zeros_like(image)
    # 红色表示需要inpaint的区域 (值=0)
    mask_vis[mask == 0] = [0, 0, 255]  # 红色
    # 绿色表示有几何信息的区域 (值>0)
    mask_vis[mask > 0] = [0, 255, 0]  # 绿色
    
    # 叠加到原图
    overlay = cv2.addWeighted(image, 0.7, mask_vis, 0.3, 0)
    
    cv2.imwrite("mask_analysis_overlay.png", overlay)
    cv2.imwrite("mask_visualization.png", mask_vis)
    
    print(f"\n💾 保存了可视化结果:")
    print(f"  mask_analysis_overlay.png - 原图+mask叠加")
    print(f"  mask_visualization.png - mask可视化")

if __name__ == "__main__":
    mask_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/testOutput/frames/mask_000.png"
    image_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/testOutput/frames/frame_000.png"
    
    analyze_mask(mask_path, image_path)