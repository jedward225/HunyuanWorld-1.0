import cv2
import numpy as np
import os
import sys
from .pano_tool import Equirec2Perspec as E2P
from .pano_tool import multi_Perspec2Equirec as m_P2E
import uuid
from PIL import Image
from tqdm import tqdm


def video2pano(image_paths,out_dir,fov=90):
    
    pers = [cv2.imread(image_path) for image_path in image_paths]
    
    # 原有8角度实现（注释掉）
    # ee = m_P2E.Perspective(pers,
    #                         [[fov, 0, 0], [fov, 45, 0], [fov, 90, 0], [fov, 135, 0],
    #                          [fov, 180, 0], [fov, 225, 0], [fov, 270, 0], [fov, 315, 0]]
    #                         )
    
    # 原有的12角度版本（注释掉）
    # if fov <= 70:  # 对于小FOV（如67.5度），使用更密集的采样
    #     num_angles = 12
    #     interval = 30  # 30度间隔，保证充分重叠
    #     print(f"Using dense sampling for FOV={fov}°: {num_angles} angles, {interval}° interval")
    # else:  # 对于大FOV（如90度），使用原来的8个角度
    #     num_angles = 8  
    #     interval = 45  # 45度间隔
    #     print(f"Using standard sampling for FOV={fov}°: {num_angles} angles, {interval}° interval")
    
    # 新的优化版本：匹配90度FOV的50%重叠率
    if fov <= 70:  # 对于小FOV（如67.5度）
        num_angles = 10
        interval = 36  # 36度间隔，重叠率31.5度（接近50%）
        print(f"Using 50% overlap sampling for FOV={fov}°: {num_angles} angles, {interval}° interval")
    else:  # 对于大FOV（如90度），使用原来的8个角度
        num_angles = 8  
        interval = 45  # 45度间隔
        print(f"Using standard sampling for FOV={fov}°: {num_angles} angles, {interval}° interval")
    
    # 生成角度列表
    angles = [i * interval for i in range(num_angles)]
    print(f"Required angles: {angles}")
    
    # 检查image_paths数量是否匹配
    if len(image_paths) != len(angles):
        print(f"Warning: Expected {len(angles)} images for FOV={fov}°, but got {len(image_paths)}")
        print("Falling back to original 8-angle method")
        # 如果数量不匹配，使用原来的固定8角度方式
        ee = m_P2E.Perspective(pers,
                                [[fov, 0, 0], [fov, 45, 0], [fov, 90, 0], [fov, 135, 0],
                                 [fov, 180, 0], [fov, 225, 0], [fov, 270, 0], [fov, 315, 0]]
                                )
    else:
        # 构建相机参数列表：[fov, yaw, pitch]
        camera_params = [[fov, angle, 0] for angle in angles]
        print(f"Using camera params: {camera_params}")
        ee = m_P2E.Perspective(pers, camera_params)

    new_pano = ee.GetEquirec(2048, 4096)
    resized_pano = cv2.resize(new_pano[650:-650], (1280, 576), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(out_dir, 'pano.png'), resized_pano.astype(np.uint8))
    return new_pano