# PanoFlexWorld: Towards World Model with Consistent 3D Inpainting

[![HunyuanWorld](https://img.shields.io/badge/HunyuanWorld-1.0-blue.svg)](https://github.com/Tencent/HunyuanWorld)
[![FlexWorld](https://img.shields.io/badge/FlexWorld-Integration-green.svg)](https://github.com/ml-gsai/FlexWorld)

## 📋 Project Overview

**PanoFlexWorld** is a pipeline that transforms text prompts or images into fully navigable 3D virtual worlds. By integrating Tencent's HunyuanWorld panorama generation with our own FlexWorld's 3D scene expansion capabilities, users can generate immersive 3D environments and freely explore them from any viewpoint.

## 🔬 Technical Approach

We leverage the power of two cutting-edge technologies:

1. **HunyuanWorld**: Tencent's powerful text/image-to-panorama model that generates high-quality 360° equirectangular panoramas (1920×960)
2. **FlexWorld**: Advanced 3D scene expansion framework using Gaussian Splatting for flexible view synthesis

### Pipeline Architecture

```mermaid
graph LR
    A[Text/Image Input] --> B[HunyuanWorld]
    B --> C[360° Panorama]
    C --> D[MoGe Depth Estimation]
    D --> E[Depth Map]
    C & E --> F[Point Cloud Generation]
    F --> G[3D Point Cloud]
    G --> H[FlexWorld Rendering]
    H --> I[Navigable 3D World]
```

### Technical Details

#### 1. Panorama Generation

- Converts text prompts or images into 360° equirectangular panoramas
- Fixed resolution: 1920×960 pixels
- Uses FLUX.1-dev base model with HunyuanWorld LoRA weights

#### 2. Depth Estimation

- Employs MoGe model for monocular depth estimation
- Generates per-pixel depth values for the entire panorama
- Outputs depth map in `.npy` format with visualization

#### 3. Point Cloud Construction

- Converts panorama + depth into 3D point cloud
- Spherical to Cartesian coordinate transformation
- Each pixel becomes a 3D point with RGB color
- Outputs in PLY/PCD format (~1.8M points for full resolution)

#### 4. 3D Rendering & Navigation

- Uses Gaussian Splatting for real-time rendering
- Supports flexible camera trajectories (orbit, forward, custom paths)
- Generates navigable videos and reconstructed panoramas

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies (see INSTALL.md for details)
conda create -n hunyuanworld python=3.9
conda activate hunyuanworld

# Install required packages
git clone https://github.com/jedward225/HunyuanWorld-1.0.git
cd HunyuanWorld-1.0
conda env create -f docker/HunyuanWorld.yaml

# real-esrgan install
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install basicsr-fixed
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop

# zim anything install & download ckpt from ZIM project page
cd ..
git clone https://github.com/naver-ai/ZIM.git
cd ZIM; pip install -e .
mkdir zim_vit_l_2092
cd zim_vit_l_2092
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx

# TO export draco format, you should install draco first
cd ../..
git clone https://github.com/google/draco.git
cd draco
mkdir build
cd build
cmake ..
make
sudo make install

# login your own hugging face account
cd ../..
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

### One-Command Pipeline

We provide a unified script `pipeline.sh` that automates the entire process:

```bash
# Examples (You can use `--help` for help):
# 1. 从文本生成：
./pipeline.sh --mode text --input "a beautiful street scene with buildings" --scene street
# 2. 从图片生成：
./pipeline.sh --mode image --input "/path/to/your/image.jpg" --scene custom
# 3. 使用已有全景图：
./pipeline.sh --mode existing --input "/path/to/panorama.png" --scene existing_pano
# 4. 只调整已有全景图尺寸：
./pipeline.sh --mode existing --input "/path/to/panorama.png" --scene resized --resize-only
# 5. 指定GPU：
./pipeline.sh --mode text --input "mountain landscape" --scene mountain --gpu 0
```

## 📁 Project Structure

```
HunyuanWorld-1.0/
├── pipeline.sh                  # 🎯 Main pipeline script
├── demo_panogen_local.py        # Panorama generation
├── generate_pano_depth.py       # Depth estimation
├── generate_pano_pointcloud.py  # Point cloud generation
├── ljj.sh                       # FlexWorld rendering script
├── FlexWorld/
│   ├── ljj.py                   # Core integration code
│   └── testOutput/              # Rendered outputs
│       ├── frames/              # 🆕 Individual frames & masks
│       │   ├── frame_000.png    # RGB frames (512×512)
│       │   ├── mask_000.png     # Coverage masks
│       │   └── inpaint_mask_000.png  # Inpainting masks
│       ├── test_video.mp4       # 360° orbit video
│       ├── test_video_mask.mp4  # Mask video
│       └── panorama_output/     # Reconstructed panorama
│           ├── pano.png         # Full panorama (1280×576)
│           ├── mask_for_inpainting.png
│           └── inpaint_mask.png
└── test_results/                # Generated assets
    └── [scene_name]/
        ├── panorama.png         # Generated panorama
        ├── depth/               # Depth maps
        └── pointcloud/          # 3D point clouds
```

## 🔧 Pipeline Components

### 1. Text/Image → Panorama (`demo_panogen_local.py`)

```bash
python demo_panogen_local.py \
    --prompt "mountain landscape with snow" \
    --output_path test_results/mountain \
    --seed 42 \
    --use_local
```

### 2. Panorama → Depth (`generate_pano_depth.py`)

```bash
python generate_pano_depth.py \
    --image_path test_results/mountain/panorama.png \
    --output_path test_results/mountain/depth \
    --verbose
```

### 3. Panorama + Depth → Point Cloud (`generate_pano_pointcloud.py`)

```bash
python generate_pano_pointcloud.py \
    --rgb_path test_results/mountain/panorama.png \
    --depth_path test_results/mountain/depth/panorama_depth.npy \
    --output_path test_results/mountain/pointcloud
```

### 4. Point Cloud → 3D World (`FlexWorld/ljj.py`)

```bash
cd FlexWorld
python ljj.py  # Automatically loads the point cloud
```

## 🎮 Camera Controls

The system supports various camera trajectories:

- **360° Orbit**: Circular path around the scene
- **Forward Motion**: Move into the scene
- **Custom Paths**: Define your own trajectory

```python
# In FlexWorld/ljj.py
traj_orbit = plan.add_traj().move_orbit_to(0, 360, 0.5, num_frames=72).finish()
```

## 🎯 Mask Generation for Inpainting

Our pipeline automatically generates inpainting masks to identify regions that lack 3D information:

### How Point Cloud Coverage is Determined

The system uses **Gaussian Splatting's alpha channel** to determine point cloud coverage:

```python
# During rendering with mask=True
rgb, depth_img, alpha_img, *_ = gs.render(cam)
if mask:
    return alpha_img  # Returns transparency/coverage information
```

**Alpha Channel Interpretation**:

- **Alpha = 1.0**: Full opacity, sufficient point cloud coverage
- **Alpha = 0.0**: Full transparency, no point cloud coverage (needs inpainting)
- **Alpha $\in$ (0,1)**: Partial coverage, sparse point cloud

### Generated Mask Files

The pipeline produces mask outputs at both panorama and individual frame levels:

#### Panorama Level:

1. **`pano.png`** - Original reconstructed panorama (1280×576)
2. **`mask_for_inpainting.png`** - Coverage visualization (white = valid, black = missing)
3. **`inpaint_mask.png`** - Inpainting-ready mask (black = regions to fill)

#### Individual Frame Level:

Located in `testOutput/frames/` directory:

1. **`frame_XXX.png`** - Individual RGB frames (512×512) from 360° orbit
2. **`mask_XXX.png`** - Per-frame coverage masks (white = valid, black = missing)
3. **`inpaint_mask_XXX.png`** - Per-frame inpainting masks (black = fill, white = keep)

### Coverage Analysis

```bash
# Example output from pipeline
Coverage: 87.3% (1610394/1843200 pixels)
```

### Why Gaps Occur

**Root Cause**: Depth discontinuities in the original panorama

- Large depth differences between adjacent pixels create spatial gaps
- Camera rotation reveals previously occluded empty regions
- Point cloud sparsity in areas with complex geometry

**Common Gap Locations**:

- Object boundaries (foreground/background transitions)
- Areas behind foreground objects
- Regions with insufficient depth information
- Edge artifacts from depth estimation

## 🐛 Known Issues & Solutions

### Black Regions in Rendered Videos

**Cause**: Large depth discontinuities create gaps when rotating the camera

**Current Solutions**:

- Coordinate system alignment (90° X-axis, -90° Y-axis rotation)
- Adjusted camera trajectory parameters
- **Automatic mask generation for targeted inpainting**

**Future Work**:

- Implement inpainting for gap filling using generated masks
- Sky layer separation for unbounded scenes
- Depth-aware point cloud densification

### Panorama Reconstruction Quality

**Issue**: FlexWorld's `video2pano` expects specific perspective views

**Solution**: Select 8 key frames from orbit video at correct angles (0°, 45°, 90°, etc.)

## 🎮 Post-Processing Workflows

### Frame-by-Frame Inpainting

```bash
# Process individual frames with highest coverage
python inpaint_single_frame.py --frame testOutput/frames/frame_036.png \
                               --mask testOutput/frames/inpaint_mask_036.png
```

### Batch Processing

```bash
# Process all frames automatically
for i in {000..071}; do
  python inpaint_frame.py --frame testOutput/frames/frame_$i.png \
                         --mask testOutput/frames/inpaint_mask_$i.png
done
```

### Quality Analysis

```bash
# Analyze coverage statistics per frame
python analyze_coverage.py --frames_dir testOutput/frames/
```

## 🚀 Incremental RGBD Completion Pipeline

### 💡 Core Innovation: 3D-Consistent Progressive Scene Completion

We are implementing an advanced **incremental RGBD completion system** that maintains 3D consistency across camera movements by progressively updating the point cloud representation.

#### 🎯 Technical Approach

**Problem**: When camera orbits around the scene, some viewpoints reveal regions with no point cloud coverage (black holes in rendered frames).

**Solution**: Frame-by-frame RGBD completion with incremental 3D scene updates:

```mermaid
graph LR
    A[Frame N] --> B[Detect Missing Regions]
    B --> C[FLUX RGB Completion]
    C --> D[Depth Estimation]
    D --> E[FlexWorld Depth Alignment]
    E --> F[Reproject to 3D Points]
    F --> G[Update Point Cloud]
    G --> H[Frame N+1 Benefits]
    H --> A
```

#### 🔧 Implementation Strategy

**Phase 1: Single Frame RGBD Completion**

- **RGB Completion**: Use FLUX.1-dev-Controlnet-Inpainting-Alpha (local model: `/mnt2/FLUX.1-dev-Controlnet-Inpainting-Alpha`) for high-quality inpainting of missing pixels
- **Depth Estimation**: Apply MoGe or similar model to completed RGB regions
- **Depth Alignment**: Use FlexWorld's `refine_depth2()` to blend new depth with existing depth
- **Quality Assessment**: Validate completion quality before 3D projection

**Phase 2: 3D Consistency Through Incremental Updates**

- **Point Cloud Projection**: Convert completed RGBD to 3D points using existing `depth2pcd_world()`
- **Incremental Addition**: Add new points to scene using existing `pcd.add_pts()` mechanism
- **Outlier Filtering**: Apply existing `remove_outliers()` for quality control
- **Progressive Coverage**: Each completed frame improves coverage for subsequent frames

#### 📋 Pipeline Workflow

```python
# Existing FlexWorld infrastructure we build upon:
for frame_i in orbit_trajectory:
    # 1. Render current view
    rgb_frame = pcd.render(cam_i)
    mask = pcd.render(cam_i, mask=True)  # Alpha channel indicates coverage
    depth_frame = pcd.render(cam_i, depth=True)
    
    # 2. RGBD Completion (NEW)
    completed_rgb = flux_inpaint(rgb_frame, mask, prompt)
    estimated_depth = depth_model(completed_rgb)
    refined_depth = refine_depth2(depth_frame, estimated_depth, mask)
    
    # 3. 3D Update (Leveraging existing FlexWorld functions)
    new_points_3d = depth2pcd_world(refined_depth, cam_i)  # EXISTING
    new_points_6d = concat([new_points_3d, completed_rgb], axis=-1)
    pcd.add_pts(new_points_6d[mask])  # EXISTING
    
    # 4. Next frame benefits from updated point cloud
```

#### 🎮 Key Advantages

1. **3D Consistency**: Unlike independent frame inpainting, maintains spatial coherence
2. **Progressive Improvement**: Each frame completion benefits subsequent frames  
3. **Leverages Existing Code**: Built on FlexWorld's proven `view_extend.py` framework
4. **Quality Control**: Multiple validation steps ensure robust completion
5. **Flexible Integration**: Can work with different inpainting and depth models

#### 🛠️ Technical Components

**Existing FlexWorld Infrastructure** (We leverage):

- ✅ `depth2pcd_world()`: RGBD → 3D point conversion  
- ✅ `refine_depth2()`: Depth map alignment with least-squares scaling
- ✅ `pcd.add_pts()`: Incremental point cloud updates
- ✅ `remove_outliers()`: Point cloud quality control
- ✅ Camera trajectory and rendering system

**New RGBD Completion Module** (We implement):

- 🔄 FLUX.1-dev integration for RGB completion
- 🔄 Depth estimation model integration  
- 🔄 Quality assessment and validation
- 🔄 Batch processing for full orbit sequences
- 🔄 Progress tracking and visualization

#### 📊 Expected Performance

- **FLUX RGB Completion**: ~15-27 seconds per frame (depends on CFG settings)
- **Depth Estimation**: ~2-5 seconds per frame  
- **3D Point Integration**: ~1 second per frame
- **72-frame Orbit Completion**: ~20-35 minutes total
- **Memory Usage**: Incremental, scales with scene complexity

#### 🎯 Validation Strategy

1. **Visual Quality**: Compare completed frames with ground truth (if available)
2. **3D Consistency**: Verify point cloud coherence across viewpoints
3. **Temporal Consistency**: Check for artifacts between consecutive frames
4. **Coverage Improvement**: Measure reduction in black regions over time

## 🔮 Future Enhancements

1. **Advanced Depth Models**
   - Integration with state-of-the-art monocular depth estimation
   - Multi-view depth consistency enforcement
   - Uncertainty-aware depth completion

2. **Selective Processing**
   - Identify and prioritize frames with best coverage
   - Quality-based frame selection for inpainting
   - Temporal consistency across frames

3. **Sky Handling**
   - Separate sky layer processing
   - Infinite depth modeling
   - Multi-layer rendering

4. **Real-time Navigation**
   - Interactive 3D viewer
   - VR/AR support
   - Dynamic scene updates

## 📊 Performance

- **Panorama Generation**: ~xx seconds
- **Depth Estimation**: ~xx seconds  
- **Point Cloud Generation**: ~xx seconds
- **3D Rendering**: ~xx seconds for 72-frame video
- **Total Pipeline**: ~x minute

## 🤝 Acknowledgments

This project integrates:

- [HunyuanWorld](https://github.com/Tencent/HunyuanWorld) by Tencent
- [FlexWorld](https://github.com/ml-gsai/FlexWorld) by ML-GSAI
- [MoGe](https://github.com/microsoft/MoGe) for depth estimation
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for rendering

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{panoflexworld2025,
  title={PanoFlexWorld: Towards World Model with Consistent 3D Inpainting},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/PanoFlexWorld}
}
```

## 📜 License

This project is released under the MIT License. See LICENSE file for details.

---

*Last Updated: 2025-08-12*
*Status: Active Development*