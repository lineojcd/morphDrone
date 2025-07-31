import OpenEXR
import Imath
import matplotlib.pyplot as plt
import numpy as np
import os

# Set parameters for showing images and histograms
# SHOW_EXR= True
SHOW_EXR = False
SHOW_HISTOGRAM = False
SHOW_FILE_LIST = False
LIDAR_REGION = (24, 32)


jobfolder = '/Users/jcd/BlenderProjects/output/Depth'
# Get all files from jobfolder
exr_files = [os.path.join(jobfolder, f) for f in os.listdir(jobfolder) if os.path.isfile(os.path.join(jobfolder, f))]
exr_files.sort()

if SHOW_FILE_LIST:
    print(f"Files in jobfolder: {exr_files}")

lidar_list = []
job_count = 0
for exr in  exr_files:
    job_count += 1
    
    print(f"Job {job_count},  processing file: {exr}")
    exr_file = OpenEXR.InputFile(exr)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr_file.channel('R', pt)
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])
    print(f"Depth shape: {depth.shape}\nMax depth value: {np.max(depth)}\nMin depth value: {np.min(depth)}")
    median_depth = np.median(depth)
    print(f"Median depth value of whole image: {median_depth}")

    # Simulate a Lidar region extraction by cropping the center of the depth image
    # Extract the center 64x48 region of the depth image
    center_h, center_w = LIDAR_REGION
    h, w = depth.shape
    start_y = (h - center_h) // 2
    start_x = (w - center_w) // 2
    center_depth = depth[start_y:start_y+center_h, start_x:start_x+center_w]

    # Show statistics of the Lidar region
    print(f"Lidar Depth shape: {center_depth.shape}\nMax depth value: {np.max(center_depth)}\nMin depth value: {np.min(center_depth)}")
    median_center_depth = np.median(center_depth)
    print(f"Median depth value of Lidar region: {median_center_depth}")
    lidar_list.append(median_center_depth)


print("length of lidar_list:", len(lidar_list))

plt.figure(figsize=(8, 5))
plt.hist(lidar_list, bins=100, range=(0, 4))

# Add value labels on top of each bar, showing the depth value (bin center)
counts, bins, _ = plt.hist(lidar_list, bins=100, range=(0, 4), alpha=0)  # Do not draw again
for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
    if count > 0:
        bin_center = (bin_left + bin_right) / 2
        plt.text(bin_center, count, f"{bin_center:.2f}", ha='center', va='bottom', fontsize=7, rotation=90, color='blue')

plt.xlabel('Depth Value')
plt.ylabel('Count')
plt.title('Histogram of Lidar Region Median Depths')
plt.xlim(0, 4)
plt.show()



