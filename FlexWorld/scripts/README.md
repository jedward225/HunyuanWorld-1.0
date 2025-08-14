# FlexWorld Scripts

从 `incremental_pipeline.py` 中提取的独立脚本，便于调试和单独测试。

## 📋 脚本列表

### 1. coordinate_transform.py
**功能**: 对点云应用坐标系变换

**用法**:
```bash
python coordinate_transform.py \
    --input /path/to/input.ply \
    --output /path/to/output.ply \
    --angle_x 90 --angle_y -90 --angle_z 0
```

### 2. render_frame.py  
**功能**: 从点云渲染单帧图像

**用法**:
```bash
python render_frame.py \
    --pointcloud /path/to/pointcloud.ply \
    --output_dir /path/to/frames \
    --frame_idx 0 \
    --cam_R '[R_matrix_as_json]' \
    --cam_T '[T_vector_as_json]' \
    --cam_f 256.0 \
    --cam_c '[256, 256]'
```

### 3. inpaint_frame.py
**功能**: 使用FLUX对单帧进行RGB补全

**用法**:
```bash
conda activate flux-inpainting
python inpaint_frame.py \
    --rgb_path /path/to/frame.png \
    --mask_path /path/to/mask.png \
    --output_path /path/to/inpainted.png \
    --frame_idx 0
```

### 4. update_pointcloud.py
**功能**: 将补全的像素反投影为3D点并更新点云

**用法**:
```bash
python update_pointcloud.py \
    --current_pointcloud /path/to/current.ply \
    --inpainted_rgb /path/to/inpainted.png \
    --original_depth /path/to/depth.npy \
    --mask_path /path/to/mask.png \
    --output_pointcloud /path/to/updated.ply \
    --frame_idx 0 \
    --cam_R '[R_matrix_as_json]' \
    --cam_T '[T_vector_as_json]' \
    --cam_f 256.0 \
    --cam_c '[256, 256]' \
    --debug
```

## 🔧 调试用法

现在可以单独运行每个步骤来定位问题：

```bash
# 1. 坐标变换
python scripts/coordinate_transform.py --input street_pointcloud.ply --output transformed.ply

# 2. 渲染测试
python scripts/render_frame.py --pointcloud transformed.ply --output_dir frames --frame_idx 0 [相机参数]

# 3. 补全测试  
python scripts/inpaint_frame.py --rgb_path frames/frame_000.png --mask_path frames/mask_000.png --output_path inpainted.png --frame_idx 0

# 4. 点云更新测试
python scripts/update_pointcloud.py --current_pointcloud transformed.ply [其他参数] --debug
```

## 🎯 优势

- **独立调试**: 每个步骤可单独测试
- **参数透明**: 所有参数都显式传递
- **环境隔离**: 明确哪个脚本需要哪个conda环境
- **日志清晰**: 便于定位具体问题所在步骤