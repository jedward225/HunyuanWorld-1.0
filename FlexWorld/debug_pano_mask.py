#!/usr/bin/env python3
"""
Debug panorama mask values to understand the distribution
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load both masks
inpaint_mask_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/testOutput/panorama_output/inpaint_mask.png"
coverage_mask_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/testOutput/panorama_output/mask_for_inpainting.png"

inpaint_mask = cv2.imread(inpaint_mask_path, cv2.IMREAD_GRAYSCALE)
coverage_mask = cv2.imread(coverage_mask_path, cv2.IMREAD_GRAYSCALE)

print("=== Panorama Mask Analysis ===")
print(f"Inpaint mask shape: {inpaint_mask.shape}")
print(f"Coverage mask shape: {coverage_mask.shape}")

print(f"\n=== inpaint_mask.png (black=need inpaint) ===")
print(f"Min value: {inpaint_mask.min()}")
print(f"Max value: {inpaint_mask.max()}")
print(f"Mean value: {inpaint_mask.mean():.2f}")
print(f"Unique values (first 20): {np.unique(inpaint_mask)[:20]}")

print(f"\n=== mask_for_inpainting.png (white=has coverage) ===")
print(f"Min value: {coverage_mask.min()}")
print(f"Max value: {coverage_mask.max()}")
print(f"Mean value: {coverage_mask.mean():.2f}")
print(f"Unique values (first 20): {np.unique(coverage_mask)[:20]}")

# Count pixels at different thresholds for inpaint_mask
total_pixels = np.prod(inpaint_mask.shape)
print(f"\n=== Inpaint Mask Threshold Analysis ===")
print(f"Total pixels: {total_pixels}")

for threshold in [0, 10, 20, 30, 50, 100, 128]:
    count = np.sum(inpaint_mask <= threshold)
    percentage = count / total_pixels * 100
    print(f"Pixels <= {threshold}: {count} ({percentage:.2f}%)")

# Show histograms
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].hist(inpaint_mask.flatten(), bins=50, alpha=0.7, color='red')
axes[0].set_title('inpaint_mask.png Distribution\n(black=need inpaint)')
axes[0].set_xlabel('Pixel Value')
axes[0].set_ylabel('Count')

axes[1].hist(coverage_mask.flatten(), bins=50, alpha=0.7, color='blue')
axes[1].set_title('mask_for_inpainting.png Distribution\n(white=has coverage)')
axes[1].set_xlabel('Pixel Value')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('panorama_mask_histogram.png', dpi=150, bbox_inches='tight')
plt.close()

# Create threshold comparison for inpaint_mask
thresholds = [0, 10, 20, 50]
fig, axes = plt.subplots(1, len(thresholds), figsize=(20, 5))

for i, thresh in enumerate(thresholds):
    binary_mask = (inpaint_mask <= thresh).astype(np.uint8) * 255
    axes[i].imshow(binary_mask, cmap='gray')
    axes[i].set_title(f'inpaint_mask <= {thresh}\n{np.sum(inpaint_mask <= thresh)} pixels')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('panorama_mask_thresholds.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ Analysis complete!")
print("📊 Check panorama_mask_histogram.png and panorama_mask_thresholds.png")