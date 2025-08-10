#!/bin/bash

# Configuration
SCENE="street"  # Change this to: demo, street, design, snowmnt, livingroom, beach_sunset, handrail

# Set the point cloud path
PLY_PATH="/home/liujiajun/HunyuanWorld-1.0/test_results/${SCENE}/pointcloud/panorama_pointcloud.ply"

# Update the path in ljj.py
sed -i "s|pcd=PcdMgr(ply_file_path=f'.*')|pcd=PcdMgr(ply_file_path=f'${PLY_PATH}')|g" FlexWorld/ljj.py

echo "Updated PLY path to: ${PLY_PATH}"

# Run the script
cd FlexWorld
python ljj.py