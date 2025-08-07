#!/usr/bin/env python3
"""
Panoramic Point Cloud Generation Script
Generate 3D point cloud from panoramic RGB image and depth map
"""

import os
import numpy as np
import cv2
import argparse
import open3d as o3d
from PIL import Image
import torch


def spherical_to_cartesian(theta, phi, depth):
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        theta: Azimuth angle in radians (horizontal, 0 to 2π)
        phi: Elevation angle in radians (vertical, -π/2 to π/2)
        depth: Distance from origin
    
    Returns:
        x, y, z: Cartesian coordinates
    """
    x = depth * np.cos(phi) * np.cos(theta)
    y = depth * np.cos(phi) * np.sin(theta)
    z = depth * np.sin(phi)
    return x, y, z


def panorama_to_pointcloud(
    rgb_path: str,
    depth_path: str,
    output_path: str,
    scale: float = 1.0,
    downsample_factor: int = 1,
    min_depth: float = 0.1,
    max_depth: float = 100.0,
    verbose: bool = True
):
    """
    Generate 3D point cloud from panoramic RGB image and depth map.
    
    Args:
        rgb_path: Path to the panoramic RGB image
        depth_path: Path to the depth map (.npy file or image)
        output_path: Directory to save the point cloud
        scale: Scale factor for depth values
        downsample_factor: Downsample factor for point cloud (1 = no downsampling)
        min_depth: Minimum depth threshold
        max_depth: Maximum depth threshold
        verbose: Whether to print detailed information
    """
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    if verbose:
        print(f"🖼️  Loading panoramic RGB image from: {rgb_path}")
    
    # Load RGB image
    rgb_image = cv2.imread(rgb_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]
    
    if verbose:
        print(f"📐 Image size: {width}x{height}")
        print(f"📊 Loading depth map from: {depth_path}")
    
    # Load depth map
    if depth_path.endswith('.npy'):
        depth_map = np.load(depth_path)
    else:
        # Load depth image (grayscale)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_image is None:
            depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth_map = depth_image.astype(np.float32)
    
    # Apply scale
    depth_map = depth_map * scale
    
    if verbose:
        print(f"📈 Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
    
    # Downsample if requested
    if downsample_factor > 1:
        height_ds = height // downsample_factor
        width_ds = width // downsample_factor
        rgb_image = cv2.resize(rgb_image, (width_ds, height_ds))
        depth_map = cv2.resize(depth_map, (width_ds, height_ds))
        height, width = height_ds, width_ds
        if verbose:
            print(f"⬇️  Downsampled to: {width}x{height}")
    
    # Create meshgrid for pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert pixel coordinates to spherical coordinates
    # u (horizontal) -> theta (azimuth): 0 to 2π
    # v (vertical) -> phi (elevation): -π/2 to π/2
    theta = (u / width) * 2 * np.pi  # 0 to 2π
    phi = ((v / height) - 0.5) * np.pi  # -π/2 to π/2
    
    # Flatten arrays
    theta_flat = theta.flatten()
    phi_flat = phi.flatten()
    depth_flat = depth_map.flatten()
    rgb_flat = rgb_image.reshape(-1, 3)
    
    # Filter by depth range
    valid_mask = (depth_flat > min_depth) & (depth_flat < max_depth) & np.isfinite(depth_flat)
    
    if verbose:
        print(f"✅ Valid points: {valid_mask.sum()} / {len(valid_mask)} ({100*valid_mask.sum()/len(valid_mask):.1f}%)")
    
    # Apply mask
    theta_valid = theta_flat[valid_mask]
    phi_valid = phi_flat[valid_mask]
    depth_valid = depth_flat[valid_mask]
    rgb_valid = rgb_flat[valid_mask]
    
    # Convert to Cartesian coordinates
    x, y, z = spherical_to_cartesian(theta_valid, phi_valid, depth_valid)
    
    # Stack coordinates
    points = np.stack([x, y, z], axis=-1)
    
    # Normalize RGB to [0, 1]
    colors = rgb_valid / 255.0
    
    # Create Open3D point cloud
    if verbose:
        print("🔨 Creating Open3D point cloud...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals
    if verbose:
        print("📐 Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Save point cloud
    output_ply = os.path.join(output_path, "panorama_pointcloud.ply")
    output_pcd = os.path.join(output_path, "panorama_pointcloud.pcd")
    
    o3d.io.write_point_cloud(output_ply, pcd)
    o3d.io.write_point_cloud(output_pcd, pcd)
    
    if verbose:
        print("💾 Saved point cloud files:")
        print(f"   - PLY format: {output_ply}")
        print(f"   - PCD format: {output_pcd}")
        print(f"📊 Point cloud statistics:")
        print(f"   - Total points: {len(pcd.points)}")
        print(f"   - Bounding box: {pcd.get_min_bound()} to {pcd.get_max_bound()}")
    
    # Optional: Create a mesh using Poisson reconstruction
    # if verbose:
    #     print("🔺 Attempting Poisson surface reconstruction...")
    
    # try:
    #     mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #         pcd, depth=9, width=0, scale=1.1, linear_fit=False
    #     )
        
    #     # Remove outliers
    #     mesh.remove_degenerate_triangles()
    #     mesh.remove_duplicated_triangles()
    #     mesh.remove_duplicated_vertices()
    #     mesh.remove_non_manifold_edges()
        
    #     # Save mesh
    #     output_mesh = os.path.join(output_path, "panorama_mesh.ply")
    #     o3d.io.write_triangle_mesh(output_mesh, mesh)
        
    #     if verbose:
    #         print(f"   - Mesh saved: {output_mesh}")
    #         print(f"   - Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")
    # except Exception as e:
    #     if verbose:
    #         print(f"   ⚠️ Mesh reconstruction failed: {str(e)}")
    
    return pcd


def visualize_pointcloud(pcd):
    """
    Visualize the point cloud using Open3D viewer.
    
    Args:
        pcd: Open3D point cloud object
    """
    print("🎨 Opening 3D viewer...")
    print("   Controls:")
    print("   - Mouse: Rotate view")
    print("   - Scroll: Zoom")
    print("   - R: Reset view")
    print("   - Q/ESC: Close viewer")
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Panoramic Point Cloud",
        width=1280,
        height=720,
        left=50,
        top=50,
        point_show_normal=False
    )


def main():
    parser = argparse.ArgumentParser(description="Generate 3D point cloud from panoramic RGB and depth")
    parser.add_argument("--rgb_path", type=str, required=True,
                        help="Path to the panoramic RGB image")
    parser.add_argument("--depth_path", type=str, required=True,
                        help="Path to the depth map (.npy file or image)")
    parser.add_argument("--output_path", type=str, default="pointcloud_output",
                        help="Directory to save the point cloud")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for depth values")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample factor (1 = no downsampling)")
    parser.add_argument("--min_depth", type=float, default=0.1,
                        help="Minimum depth threshold")
    parser.add_argument("--max_depth", type=float, default=100.0,
                        help="Maximum depth threshold")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the point cloud after generation")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed information")
    
    args = parser.parse_args()
    
    print("🌟 Panoramic Point Cloud Generation")
    print("=" * 50)
    
    try:
        pcd = panorama_to_pointcloud(
            rgb_path=args.rgb_path,
            depth_path=args.depth_path,
            output_path=args.output_path,
            scale=args.scale,
            downsample_factor=args.downsample,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            verbose=args.verbose
        )
        
        print("🎉 Point cloud generation completed successfully!")
        
        if args.visualize:
            visualize_pointcloud(pcd)
        
    except Exception as e:
        print(f"❌ Error during point cloud generation: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())