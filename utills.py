import numpy as np

def generate_moving_square(num_frames=15, frame_height=128, frame_width=128,
                            square_size=20, step=5):
    """
    Generates a grayscale video of a white square moving horizontally.
    Returns: numpy array of shape (num_frames, frame_height, frame_width)
    """
    video = np.zeros((num_frames, frame_height, frame_width), dtype=np.uint8)

    for i in range(num_frames):
        frame = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        # Square position moves right each frame
        x = 10 + i * step   # left edge of square
        y = 50               # top edge stays fixed
        
        # Make sure square doesn't go out of bounds
        x_end = min(x + square_size, frame_width)
        y_end = min(y + square_size, frame_height)
        
        frame[y:y_end, x:x_end] = 255  # white square
        video[i] = frame

    return video

import matplotlib.pyplot as plt

video = generate_moving_square()

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(video[i], cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Frame {i}')
    ax.axis('off')
plt.tight_layout()
plt.show()