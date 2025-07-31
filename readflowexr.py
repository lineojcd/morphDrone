import OpenEXR
import Imath
import matplotlib.pyplot as plt
import numpy as np

# Set parameters
SHOW_FLOW = False
SHOW_FLOW = True

exr_path = '/Users/jcd/BlenderProjects/output/Flow/Flow0150.exr'
exr_path = '/Users/jcd/BlenderProjects/output/Flow/Flow0002.exr'
exr_path = '/Users/jcd/BlenderProjects/output/Flow/Flow0003.exr'
exr_path = '/Users/jcd/BlenderProjects/output/Flow/Flow0001.exr'
exr_file = OpenEXR.InputFile(exr_path)
print("Available channels:", exr_file.header()['channels'].keys())

# Read header and get image size
dw = exr_file.header()['dataWindow']
size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

# Read flow channels 
pt = Imath.PixelType(Imath.PixelType.FLOAT)

# Read flow vectors
flow_x = np.frombuffer(exr_file.channel('X', pt), dtype=np.float32).reshape(size[1], size[0])
flow_y = np.frombuffer(exr_file.channel('Y', pt), dtype=np.float32).reshape(size[1], size[0])

# Compute flow magnitude for visualization
flow_mag = np.sqrt(flow_x**2 + flow_y**2)

if SHOW_FLOW:
    plt.figure()
    plt.imshow(flow_mag, cmap='inferno')
    plt.colorbar(label='Flow Magnitude')
    plt.title('Optical Flow Magnitude')
    plt.axis('off')
    plt.show()


