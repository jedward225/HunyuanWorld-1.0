# HunyuanWorld-FlexWorld 集成项目阶段性报告

## 项目概述

### 项目目标
将 HunyuanWorld-1.0 的全景图生成能力与 FlexWorld 的 3D 场景扩展能力进行集成，实现一个完整的 3D 场景生成和扩展 pipeline。

### 核心流程
1. **输入**: 文本提示词或初始图像
2. **全景生成**: 使用 HunyuanWorld 生成 360° 全景图
3. **深度估计**: 使用 MoGe 模型估计全景深度
4. **点云生成**: 将全景图+深度转换为 3D 点云
5. **视频渲染**: 使用 FlexWorld 沿相机轨迹渲染视频
6. **全景重建**: 从渲染视频重建不完整全景图（带 mask）
7. **迭代扩展**: 使用 inpainting 填充缺失区域，迭代扩展场景

## 已完成的工作

### 1. HunyuanWorld Pipeline 搭建 ✅
- **文件**: `demo_panogen_local.py`
- **功能**: 
  - 支持文本到全景图生成 (Text2Panorama)
  - 支持图像到全景图生成 (Image2Panorama)
  - 配置本地 FLUX.1-dev 模型路径
  - 集成 HunyuanWorld LoRA 权重

### 2. 深度估计模块 ✅
- **文件**: `generate_pano_depth.py`
- **功能**:
  - 使用 MoGe 模型进行全景深度估计
  - 生成深度图 (.npy 格式)
  - 生成深度可视化图像 (INFERNO colormap)
  - 生成有效像素 mask

### 3. 点云生成模块 ✅
- **文件**: `generate_pano_pointcloud.py`
- **功能**:
  - 球面坐标到笛卡尔坐标转换
  - RGB 全景图 + 深度图 → 3D 点云
  - 支持点云降采样
  - 输出 PLY 和 PCD 格式
  - 法线估计

### 4. FlexWorld 集成 ✅
- **文件**: `ljj.py`
- **主要功能**:
  - 加载点云并进行坐标系变换（90° X轴旋转，-90° Y轴旋转）
  - 使用 Gaussian Splatting 渲染视频
  - 生成 RGB 视频和 mask 视频
  - 相机轨迹控制（360° 环绕）

### 5. 视频到全景图转换 ✅
- **实现的功能**:
  - `frames_to_panorama_center_align()`: 中心对齐模式
  - `frames_to_panorama_scaled()`: 缩放模式
  - `extract_frames_from_video()`: 视频帧提取
  - `render_panorama_from_trajectory()`: 完整的渲染到全景流程

### 6. 技术问题解决 ✅
- 修复了 imageio 视频保存格式问题
- 处理了 mask 张量维度不匹配问题
- 实现了 `easy_save_video` 集成

## 当前存在的问题

### 1. 全景图拼接质量差 ⚠️
**问题描述**: 从渲染视频重建的全景图质量很差，存在严重的拼接问题

**可能原因**:
- 简单的列复制方法过于粗糙
- 没有利用真实的相机内参和外参
- 缺少球面投影的几何变换
- 相邻帧之间没有做颜色校正和边缘融合
- FOV 计算使用硬编码值（67.5°）而非真实相机参数

### 2. 相机参数未充分利用 ⚠️
- 相机焦距 f=200 的设置可能不合理
- HunyuanWorld 和 FlexWorld 的焦距设置未对齐
- 相机轨迹的 elevation 信息未被使用

### 3. 坐标系转换问题 ⚠️
- 需要手动旋转点云（X轴90°，Y轴-90°）才能正确对齐
- 可能存在坐标系定义不一致

## 待完成的工作

### 高优先级
1. **改进全景图拼接算法** 🔴
   - 研究更好的球面投影方法
   - 实现基于相机参数的精确投影
   - 添加图像特征点匹配和对齐
   - 考虑使用 OpenCV Stitcher 或专门的全景库

2. **焦距参数对齐** 🔴
   - 调研 HunyuanWorld 的相机模型
   - 统一 FlexWorld 和 HunyuanWorld 的焦距设置
   - 确保 FOV 计算的准确性

### 中优先级
3. **集成 image2pcd 功能** 🟡
   - 完成端到端的 pipeline 集成
   - 实现一键式从图像到渲染的流程

4. **相机轨迹优化** 🟡
   - 测试不同的相机轨迹（前进、螺旋、多层等）
   - 实现自适应轨迹规划

### 低优先级
5. **Inpainting 集成** 🟢
   - 研究 HunyuanWorld 的 inpainting 模型
   - 实现 mask 引导的内容生成
   - 迭代式场景扩展

6. **Sky 处理** 🟢
   - 处理无界场景中的天空区域
   - 可能需要特殊的渲染策略

## 文件结构
```
HunyuanWorld-1.0/
├── demo_panogen_local.py        # 全景图生成
├── generate_pano_depth.py       # 深度估计
├── generate_pano_pointcloud.py  # 点云生成
├── depth.sh                      # 深度生成脚本
├── pc.sh                        # 点云生成脚本
├── flexworld_summary_and_my_plan.md  # 项目计划
└── FlexWorld/
    ├── ljj.py                   # 主要集成代码
    ├── testOutput/              # 输出结果
    │   ├── test_video.mp4       # 渲染视频
    │   ├── test_video_mask.mp4  # mask视频
    │   └── panorama_output/
    │       ├── panorama.png     # 重建的全景图
    │       └── panorama_mask.png # 覆盖mask
    └── ...
```

## 运行示例

### 完整 Pipeline
```bash
# 1. 生成全景图
python demo_panogen_local.py --prompt "a beautiful living room" --output_path output

# 2. 生成深度图
python generate_pano_depth.py --image_path output/panorama.png --output_path output/depth

# 3. 生成点云
python generate_pano_pointcloud.py --rgb_path output/panorama.png --depth_path output/depth/panorama_depth.npy --output_path output/pointcloud

# 4. FlexWorld 渲染和全景重建
cd FlexWorld
python ljj.py  # 使用生成的点云路径
```

## 下一步行动建议

### 立即行动
1. **分析渲染视频质量**: 检查 `testOutput/test_video.mp4`，确认相机轨迹是否正确，帧之间是否有足够重叠
2. **研究现有方案**: 查看 FlexWorld 和其他项目中的全景拼接实现
3. **测试 OpenCV Stitcher**: 作为快速验证方案

### 技术调研方向
1. **球面投影几何**: 研究等距柱状投影（Equirectangular Projection）的正确实现
2. **图像配准算法**: SIFT/SURF 特征匹配，RANSAC 变换估计
3. **多带混合（Multi-band Blending）**: 解决接缝问题
4. **Bundle Adjustment**: 全局优化相机参数

## 项目亮点
- 成功打通了 HunyuanWorld → 点云 → FlexWorld 的完整流程
- 实现了基础的视频到全景图转换功能
- 解决了多个技术集成问题
- 建立了清晰的模块化架构

---
*报告生成时间: 2025-08-07*
*项目状态: 开发中*