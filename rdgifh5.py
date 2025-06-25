import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os, shutil, sys
import argparse
from matplotlib import cm

# Argument parsing for delete options
parser = argparse.ArgumentParser(description="Create a GIF from Soli HDF5 file with 2x2 channel mosaic.")
parser.add_argument("input_h5", help="Input HDF5 file")
parser.add_argument("--delete", "-d", action="store_true", help="Delete the frames directory and exit")
parser.add_argument("--cleanup", "-c", action="store_true", help="Delete the frames directory after building the GIF")
args = parser.parse_args()

file_path = args.input_h5
base_name = os.path.splitext(os.path.basename(file_path))[0]
output_gif = f'./screenshots/{base_name}.gif'
frames_dir = f'./screenshots/{base_name}.frames'

if args.delete:
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
        print(f"Deleted frames directory: {frames_dir}")
    else:
        print(f"Frames directory does not exist: {frames_dir}")
    sys.exit(0)

# Clean and prepare frames directory
if os.path.exists(frames_dir):
    shutil.rmtree(frames_dir)
os.makedirs(frames_dir, exist_ok=True)

# Load data and reshape channels to 32x32
with h5py.File(file_path, 'r') as f:
    ch_keys = sorted([k for k in f.keys() if k.startswith('ch')])
    chs = np.stack([f[k][:] for k in ch_keys], axis=1)  # shape: (frames, 4, 1024)
    num_frames = chs.shape[0]
    fps = f.attrs.get('fps', 10)  # Read FPS from HDF5, default to 10 if not present

# Set GIF frame duration (seconds per frame)
duration = 0.1  # Default GIF frame duration (as before, 0.1s per frame)

# Calculate and print timing info
real_duration = num_frames / float(fps)
gif_duration = num_frames * duration
ratio = gif_duration / real_duration if real_duration > 0 else float('inf')
print(f"Radar frames: {num_frames}")
print(f"Radar FPS (from HDF5): {fps}")
print(f"Real duration: {real_duration:.3f} s")
print(f"GIF duration: {gif_duration:.3f} s (duration per frame: {duration:.3f} s)")
print(f"GIF/Real time ratio: {ratio:.2f}x (1.0 = real speed, >1 = slower, <1 = faster)")

# For each frame, create a 2x2 mosaic of 32x32 images
for idx in range(num_frames):
    channels = chs[idx]  # (4, 1024)
    imgs = [chan.reshape(32, 32) for chan in channels]

    # Create divider (vertical and horizontal)
    divider = np.zeros((32, 2, 3), dtype=np.uint8)  # 2-pixel wide RGB bar
    divider[..., 0] = 255  # Red channel
    h_divider = np.zeros((2, 32*2 + 2, 3), dtype=np.uint8)  # 2-pixel tall RGB bar
    h_divider[..., 0] = 255  # Red channel

    # Normalize grayscale imgs to 0-255 and convert to uint8
    imgs_norm = [((img - img.min()) / (np.ptp(img) + 1e-8) * 255) for img in imgs]
    viridis = cm.get_cmap('viridis')
    imgs_rgb = [np.array(viridis(img/255.0))[..., :3] for img in imgs_norm]  # Apply colormap, drop alpha
    imgs_rgb = [(img*255).astype(np.uint8) for img in imgs_rgb]

    # Stack with dividers: [img0 | v | img1]
    top = np.hstack([imgs_rgb[0], divider, imgs_rgb[1]])
    bottom = np.hstack([imgs_rgb[2], divider, imgs_rgb[3]])
    mosaic = np.vstack([top, h_divider, bottom])

    # Save frame image
    frame_path = os.path.join(frames_dir, f'frame_{idx:03d}.png')
    plt.imsave(frame_path, mosaic, format='png')

# Assemble frames into animated GIF
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
frames = [imageio.imread(fp) for fp in frame_files]
imageio.mimsave(output_gif, frames, duration=duration, loop=0)

if args.cleanup:
    shutil.rmtree(frames_dir)
    print(f"Deleted frames directory after GIF creation: {frames_dir}")

print("Animated GIF with 32×32 channel images saved to:", output_gif)
