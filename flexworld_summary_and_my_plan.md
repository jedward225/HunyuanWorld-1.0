# FlexWorld Integration Summary and Project Plan

## FlexWorld Pipeline Overview

FlexWorld is a 3D scene generation framework that progressively expands scenes for flexible-view synthesis. It enables the creation of 360° scenes from single images using video diffusion models and 3D Gaussian Splatting.

### Core Pipeline Components

1. **Image → Point Cloud Conversion**
   - Convert input images to 3D point clouds
   - Uses depth estimation and image-to-3D techniques

2. **Point Cloud Management** 
   - `PcdMgr` class handles point cloud data
   - Supports loading from PLY files or direct numpy/tensor arrays
   - Manages 3D points with RGB colors ([N,6] format)

3. **Camera Control**
   - `CamPlanner` class defines camera trajectories
   - Supports various movement patterns (orbit, forward, etc.)
   - Flexible trajectory composition

4. **Rendering**
   - Uses Gaussian Splatting backend for rendering
   - Generates videos from point clouds along camera trajectories
   - Supports mask generation for incomplete views

## Current Implementation Analysis (ljj.py)

### Working Pipeline
- **Panorama → Point Cloud → Video rendering**
  - Loading point cloud from panorama: `/mnt/zhouzihan/BigOne/cache/panorama_pointcloud.ply`
  - Applying rotation transformations for proper orientation (90° X-axis, -90° Y-axis)
  - Rendering with customizable camera trajectories
  - Generating both RGB videos and masks

### Key Parameters
- Focal length: f=200 (adjustable, affects field of view)
- Default render backend: Gaussian Splatting ('gs')
- Output resolution: 512x512 pixels
- Frame count: 96 frames per trajectory segment

## Integration Tasks and TODOs

### High Priority
1. **Image to Point Cloud Integration**
   - Integrate `Image2Pcd_Tool` for direct image-to-3D conversion
   - Enable single Python script execution for complete pipeline

2. **Panorama Generation from Rendered Videos**
   - Implement reverse conversion: video frames → panorama
   - Generate incomplete panoramas with corresponding masks
   - This is the key output needed for the next discussion

### Medium Priority
3. **Focal Length Alignment**
   - Investigate HunyuanWorld's focal length settings
   - Ensure consistency between different pipeline components
   - Test impact of different focal lengths on output quality

4. **Camera Trajectory Exploration**
   - Test various trajectory patterns
   - Implement custom trajectories for specific use cases
   - Document optimal trajectories for different scenes

### Low Priority
5. **Sky Rendering**
   - Handle unbounded scenes with sky regions
   - May require special treatment in point cloud generation

6. **Inpainting Capabilities**
   - Investigate training-free inpainting methods
   - Could use for filling gaps in incomplete panoramas

## Pipeline Integration Strategy

### Phase 1: Basic Pipeline Setup
```python
# 1. Load input image
input_image = "path/to/image.png"

# 2. Convert to point cloud
pcd = image2pcd(input_image)

# 3. Define camera trajectory
plan = CamPlanner()
trajectory = plan.add_traj().move_orbit_to(0, 360, 0.1, num_frames=96).finish()

# 4. Render video
render_results = CamPlanner._render_video(pcd, traj=trajectory, output_path='output.mp4')
```

### Phase 2: Panorama Generation
```python
# 5. Extract frames from rendered video
frames = extract_frames(render_results)

# 6. Stitch frames into panorama
panorama, mask = frames_to_panorama(frames)

# 7. Output incomplete panorama with mask
save_panorama(panorama, mask)
```

## Next Steps

1. **Test Current Implementation**
   - Run ljj.py with existing point cloud
   - Verify video generation works correctly
   - Test different camera trajectories

2. **Implement image2pcd Function**
   - Complete the Image2Pcd_Tool integration
   - Test with sample images

3. **Develop Panorama Conversion**
   - Research frame-to-panorama stitching methods
   - Implement mask generation for incomplete regions

4. **Optimize Pipeline**
   - Tune parameters (focal length, resolution, etc.)
   - Improve processing speed
   - Add error handling

## Technical Notes

- FlexWorld uses CogVideoX-5B for video generation
- Gaussian Splatting provides efficient 3D rendering
- Point clouds are stored in PLY format with XYZRGB data
- Camera trajectories are defined as lists of Mcam objects
- Rotation transformations may be needed to align coordinate systems

## Dependencies

- **Core**: PyTorch, OmegaConf, NumPy
- **3D Processing**: gsplat, Gaussian Splatting
- **Video**: CogVideoX, save_video utilities
- **FlexWorld Modules**: PcdMgr, CamPlanner, Image2Pcd_Tool