#!/usr/bin/env python3
"""
修复深度估计问题的解决方案
"""

import numpy as np
import cv2

def improved_depth_estimation(original_depth, mask, method='scale_aware'):
    """
    改进的深度估计方法
    
    Args:
        original_depth: 原始深度图
        mask: 需要估计深度的区域
        method: 估计方法
    """
    
    if method == 'scale_aware':
        # 方法1: 基于周围有效深度的统计信息
        
        # 获取有效深度的统计信息
        valid_depths = original_depth[~mask]
        if len(valid_depths) > 0:
            depth_mean = np.mean(valid_depths)
            depth_std = np.std(valid_depths)
            depth_min = np.min(valid_depths)
            depth_max = np.max(valid_depths)
            
            print(f"📊 有效深度统计:")
            print(f"   范围: [{depth_min:.3f}, {depth_max:.3f}]")
            print(f"   均值: {depth_mean:.3f} ± {depth_std:.3f}")
            
            # 使用插值获得初始估计
            estimated_depth = cv2.inpaint(
                original_depth.astype(np.float32),
                mask.astype(np.uint8) * 255,
                inpaintRadius=15,
                flags=cv2.INPAINT_TELEA
            )
            
            # 关键：将估计深度约束到合理范围内
            # 不允许深度偏离有效范围太远
            estimated_depth = np.clip(estimated_depth, 
                                     depth_min - depth_std, 
                                     depth_max + depth_std)
            
            # 对mask区域应用额外的平滑
            kernel = cv2.getGaussianKernel(5, 1.0)
            kernel = kernel @ kernel.T
            estimated_depth_smooth = cv2.filter2D(estimated_depth, -1, kernel)
            estimated_depth[mask] = estimated_depth_smooth[mask]
            
            print(f"📊 估计深度统计:")
            print(f"   范围: [{estimated_depth[mask].min():.3f}, {estimated_depth[mask].max():.3f}]")
            
            return estimated_depth
            
    elif method == 'nearest_neighbor':
        # 方法2: 使用最近邻深度
        
        # 创建距离变换找到每个缺失像素最近的有效像素
        dist_transform = cv2.distanceTransform(
            mask.astype(np.uint8), 
            cv2.DIST_L2, 
            5
        )
        
        # 使用morphological operations获得边界深度
        kernel = np.ones((5,5), np.uint8)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        boundary = dilated_mask & (~mask.astype(np.uint8))
        
        if np.sum(boundary) > 0:
            boundary_depths = original_depth[boundary > 0]
            mean_boundary_depth = np.mean(boundary_depths)
            
            # 简单地使用边界平均深度
            estimated_depth = original_depth.copy()
            estimated_depth[mask] = mean_boundary_depth
            
            # 应用高斯平滑
            estimated_depth = cv2.GaussianBlur(estimated_depth, (7,7), 2.0)
            
            return estimated_depth
            
    elif method == 'plane_fitting':
        # 方法3: 局部平面拟合（适合建筑场景）
        
        # 将图像分成小块，每块拟合一个平面
        block_size = 32
        h, w = original_depth.shape
        estimated_depth = original_depth.copy()
        
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                
                block_depth = original_depth[y:y_end, x:x_end]
                block_mask = mask[y:y_end, x:x_end]
                
                if np.sum(block_mask) > 0 and np.sum(~block_mask) > 10:
                    # 有足够的有效点来拟合平面
                    valid_depths = block_depth[~block_mask]
                    
                    # 简单使用中值作为平面深度
                    plane_depth = np.median(valid_depths)
                    estimated_depth[y:y_end, x:x_end][block_mask] = plane_depth
        
        # 最后应用平滑
        estimated_depth = cv2.GaussianBlur(estimated_depth, (5,5), 1.5)
        
        return estimated_depth
    
    else:
        # 默认方法
        return cv2.inpaint(
            original_depth.astype(np.float32),
            mask.astype(np.uint8) * 255,
            inpaintRadius=10,
            flags=cv2.INPAINT_TELEA
        )

def scale_correction_for_new_points(new_points_3d, existing_points, scale_factor=None):
    """
    对新增点进行scale修正
    
    Args:
        new_points_3d: 新增的3D点
        existing_points: 已有的点云
        scale_factor: 手动指定的缩放因子
    """
    
    if len(new_points_3d) == 0:
        return new_points_3d
    
    # 计算已有点云的scale
    existing_scale = np.percentile(np.abs(existing_points[:, :3]), 95)  # 使用95分位数避免outlier
    new_scale = np.percentile(np.abs(new_points_3d), 95)
    
    print(f"📏 Scale分析:")
    print(f"   已有点云scale (95%): {existing_scale:.3f}")
    print(f"   新增点scale (95%): {new_scale:.3f}")
    
    if scale_factor is None:
        # 自动计算缩放因子
        if new_scale < existing_scale * 0.2:  # 新点太小
            scale_factor = existing_scale / new_scale * 0.5  # 保守一点，乘以0.5
            print(f"   ⚠️ 新点scale过小，应用放大: ×{scale_factor:.2f}")
        elif new_scale > existing_scale * 5:  # 新点太大
            scale_factor = existing_scale / new_scale * 2  # 保守一点，乘以2
            print(f"   ⚠️ 新点scale过大，应用缩小: ×{scale_factor:.2f}")
        else:
            scale_factor = 1.0
            print(f"   ✅ Scale在合理范围内")
    
    if scale_factor != 1.0:
        new_points_3d_corrected = new_points_3d * scale_factor
        print(f"   修正后scale: {np.percentile(np.abs(new_points_3d_corrected), 95):.3f}")
        return new_points_3d_corrected
    
    return new_points_3d

# 示例：如何在incremental_pipeline.py中使用
example_code = """
# 在update_pointcloud函数中修改深度估计部分：

if np.sum(mask) > 0:
    # 使用改进的深度估计
    from fix_depth_estimation import improved_depth_estimation, scale_correction_for_new_points
    
    # 使用scale-aware方法估计深度
    estimated_depth = improved_depth_estimation(
        original_depth, 
        mask, 
        method='scale_aware'
    )
    
    # ... 深度对齐等步骤 ...
    
    # 3D重建
    points_3d = depth2pcd_world(aligned_depth, cam)
    
    # 只添加mask区域的点
    new_points_3d = points_3d[mask]
    
    # 应用scale修正
    new_points_3d = scale_correction_for_new_points(
        new_points_3d, 
        current_pcd.pts
    )
    
    new_colors = inpainted_rgb[mask] / 255.0
    new_points_6d = np.concatenate([new_points_3d, new_colors], axis=-1)
"""

if __name__ == "__main__":
    print("🔧 深度估计修复方案")
    print("=" * 60)
    
    print("📋 提供了3种改进的深度估计方法:")
    print("1. scale_aware: 基于有效深度统计的约束插值")
    print("2. nearest_neighbor: 使用最近邻深度")
    print("3. plane_fitting: 局部平面拟合")
    
    print("\n📏 还提供了scale修正函数:")
    print("   自动检测并修正新增点的scale问题")
    
    print("\n💡 使用方法:")
    print(example_code)
    
    print("\n✅ 这应该能解决新增点scale过小的问题！")