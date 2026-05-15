import numpy as np

def generate_moving_square(num_frames=30, h=64, w=64, square_size=10):
    frames = np.zeros((num_frames, h, w), dtype=np.uint8)
    for i in range(num_frames):
        frame = np.zeros((h, w), dtype=np.uint8)
        x = (i * 2) % (w - square_size)
        frame[20:20+square_size, x:x+square_size] = 255
        frames[i] = frame
    return frames

def compression_ratio(original_frames, compressed_bits):
    h, w = original_frames[0].shape
    original_bits = len(original_frames) * h * w * 8

    return original_bits / compressed_bits

def mse(original, reconstructed):
    return np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)


def psnr(original, reconstructed):
    error = mse(original, reconstructed)

    if error == 0:
        return float('inf')

    return 10 * np.log10((255 ** 2) / error)