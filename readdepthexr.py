import OpenEXR
import Imath
import matplotlib.pyplot as plt
import numpy as np

# Set parameters for showing images and histograms
# SHOW_EXR= True
SHOW_EXR = False
SHOW_HISTOGRAM = False

exr_path = '/Users/jcd/BlenderProjects/output/Depth/Depth0001.exr'
exr_file = OpenEXR.InputFile(exr_path)
dw = exr_file.header()['dataWindow']
size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
pt = Imath.PixelType(Imath.PixelType.FLOAT)
depth_str = exr_file.channel('R', pt)
depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])

### Show exr image ###
# Example for reading the R channel (depth)
if SHOW_EXR:
    plt.figure()
    plt.imshow(depth, cmap='gray')
    plt.title('Depth0001.exr')
    plt.axis('off')
    plt.show()

### Show histogram of depth values ###
if SHOW_HISTOGRAM:
    plt.figure()
    plt.hist(depth.flatten(), bins=100, color='green', alpha=0.7, log=True)

    counts, bins, _ = plt.hist(depth.flatten(), bins=100, color='green', alpha=0.0, log=True)  # invisible overlay for bin info
    for i in range(len(counts)):
        if counts[i] > 1e3:
            bin_center = (bins[i] + bins[i+1]) / 2
            plt.text(bin_center, counts[i], f"{bin_center:.2f}", ha='center', va='bottom', fontsize=8, rotation=90)

    plt.title('Log Histogram of Depth Values')
    plt.xlabel('Depth Value')
    plt.ylabel('Log Count')
    plt.show()

# Show EXR image and histogram side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# Left: Depth image
im = axs[0].imshow(depth, cmap='gray')
axs[0].set_title('Depth0001.exr')
axs[0].axis('off')
fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

# Right: Histogram
counts, bins, _ = axs[1].hist(depth.flatten(), bins=100, color='green', alpha=0.7, log=True)
for i in range(len(counts)):
    if counts[i] > 1e3:
        bin_center = (bins[i] + bins[i+1]) / 2
        axs[1].text(bin_center, counts[i], f"{bin_center:.2f}", ha='center', va='bottom', fontsize=8, rotation=90)
axs[1].set_title('Log Histogram of Depth Values')
axs[1].set_xlabel('Depth Value')
axs[1].set_ylabel('Log Count')
plt.tight_layout()
plt.show()

# Show statistics of the depth image
print(depth)
print(f"Depth shape: {depth.shape}\nMax depth value: {np.max(depth)}\nMin depth value: {np.min(depth)}")
# Calculate and print the median of the center region
median_depth = np.median(depth)
print(f"Median depth value of the whole image: {median_depth}")

# Simulate a Lidar region extraction by cropping the center of the depth image
# Extract the center 64x48 region of the depth image
center_h, center_w = 48, 64
h, w = depth.shape
start_y = (h - center_h) // 2
start_x = (w - center_w) // 2
center_depth = depth[start_y:start_y+center_h, start_x:start_x+center_w]

# Show 3 images (full EXR with rectangle, center region, and center region histogram) in one plot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
# 1. Original EXR with red rectangle
im0 = axs[0].imshow(depth, cmap='gray')
rect = plt.Rectangle((start_x, start_y), center_w, center_h, linewidth=2, edgecolor='red', facecolor='none')
axs[0].add_patch(rect)
axs[0].set_title('Depth0001.exr with Center Region')
axs[0].axis('off')
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

# 2. Center region only
im1 = axs[1].imshow(center_depth, cmap='gray')
axs[1].set_title('Center 64x48 Region')
axs[1].axis('off')
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

# Add value labels on top of the histogram bars for the center region
counts, bins, _ = axs[2].hist(center_depth.flatten(), bins=50, color='orange', alpha=0.0, log=True)  # invisible overlay for bin info
for i in range(len(counts)):
    if counts[i] > 1e2:
        bin_center = (bins[i] + bins[i+1]) / 2
        axs[2].text(bin_center, counts[i], f"{bin_center:.2f}", ha='center', va='bottom', fontsize=8, rotation=90)

# 3. Histogram of center region
axs[2].hist(center_depth.flatten(), bins=50, color='orange', alpha=0.7, log=True)
axs[2].set_title('Log Histogram\nCenter Region')
axs[2].set_xlabel('Depth Value')
axs[2].set_ylabel('Log Count')
plt.tight_layout()
plt.show()

# Show statistics of the Lidar region
print(f"Depth shape: {center_depth.shape}\nMax depth value: {np.max(center_depth)}\nMin depth value: {np.min(center_depth)}")

# Calculate and print the median of the center region
median_center_depth = np.median(center_depth)
print(f"Median depth value of center region: {median_center_depth}")



