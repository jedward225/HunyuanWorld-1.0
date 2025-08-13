#!/usr/bin/env python3
"""
Debug mask values to understand the distribution
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load mask
mask_path = "/home/liujiajun/HunyuanWorld-1.0/FlexWorld/testOutput/frames/mask_009.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

print("=== Mask Analysis ===")
print(f"Mask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")
print(f"Min value: {mask.min()}")
print(f"Max value: {mask.max()}")
print(f"Unique values: {np.unique(mask)}")

# Count pixels at different thresholds
total_pixels = np.prod(mask.shape)
print(f"\nTotal pixels: {total_pixels}")

for threshold in [0, 10, 20, 50, 100, 128]:
    count = np.sum(mask <= threshold)
    percentage = count / total_pixels * 100
    print(f"Pixels <= {threshold}: {count} ({percentage:.2f}%)")

# Show histogram
plt.figure(figsize=(10, 6))
plt.hist(mask.flatten(), bins=50, alpha=0.7)
plt.title('Mask Value Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Count')
plt.savefig('mask_histogram.png')
plt.close()

# Create different threshold masks for comparison
thresholds = [0, 10, 20, 50]
fig, axes = plt.subplots(1, len(thresholds), figsize=(20, 5))

for i, thresh in enumerate(thresholds):
    binary_mask = (mask <= thresh).astype(np.uint8) * 255
    axes[i].imshow(binary_mask, cmap='gray')
    axes[i].set_title(f'Threshold <= {thresh}\n{np.sum(mask <= thresh)} pixels')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('mask_thresholds_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ Analysis complete!")
print("📊 Check mask_histogram.png and mask_thresholds_comparison.png")