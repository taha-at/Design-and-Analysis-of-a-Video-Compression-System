import numpy as np


def compute_metrics(original_frames, reconstructed_frames):
    mse_total = 0
    for orig, recon in zip(original_frames, reconstructed_frames):
        mse_total += np.mean((orig.astype(float) - recon.astype(float)) ** 2)
    mse = mse_total / len(original_frames)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    return mse, psnr

def generate_moving_square(num_frames=30, h=64, w=64, square_size=10):
    frames = []
    for i in range(num_frames):
        frame = np.zeros((h, w), dtype=np.uint8)
        x = (i * 2) % (w - square_size)
        frame[20:20+square_size, x:x+square_size] = 255
        frames.append(frame)
    return frames