import cv2
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import dahuffman
from arithmetic_coding import ArithmeticEncoder
from collections import Counter

class Encoder:
    def __init__(self, video_path, block_size, quantization_table, coding_method):
        self.video_path = video_path
        self.block_size = block_size
        self.quantization_table = quantization_table  
        self.coding_method = coding_method

        self.video = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        self.frames = frames
        self.video.release()

    def process_frame(self, frame):
        h, w = frame.shape
        pad_h = (self.block_size - (h % self.block_size)) % self.block_size
        pad_w = (self.block_size - (w % self.block_size)) % self.block_size
        padded = np.pad(frame, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        blocks = []
        for i in range(0, padded.shape[0], self.block_size):
            for j in range(0, padded.shape[1], self.block_size):
                block = padded[i:i+self.block_size, j:j+self.block_size].astype(np.float32)
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                quantized = np.round(dct_block / self.quantization_table).astype(int)
                blocks.append(quantized)

        return blocks

    def _blocks_to_symbols(self, all_frames):
        symbols = []
        for blocks in all_frames:
            for block in blocks:
                flat = block.flatten()  # zigzag would go here ideally
                run = 0
                for coeff in flat:
                    if coeff == 0:
                        run += 1
                    else:
                        symbols.append(('RLE', run, int(coeff)))
                        run = 0
                symbols.append('EOB')  # end of block
        symbols.append('<EOM>')        # end of message — required by ArithmeticEncoder
        return symbols

    def _build_frequencies(self, symbols):
        """Build frequency dict from symbols. ArithmeticEncoder needs this."""
        freq = dict(Counter(symbols))
        # Ensure <EOM> is always present
        if '<EOM>' not in freq:
            freq['<EOM>'] = 1
        return freq

    def encode(self):
        # ── Process all frames ────────────────────────────────────────────────
        all_frames = []
        for frame in self.frames:
            blocks = self.process_frame(frame)
            all_frames.append(blocks)

        # ── Huffman ───────────────────────────────────────────────────────────
        if self.coding_method == 'huffman':
            flat_blocks = np.concatenate(
                [np.array(b).flatten() for b in all_frames]
            ).flatten()
            codec = dahuffman.HuffmanCodec.from_data(flat_blocks)
            encoded_data = codec.encode(flat_blocks)
            return encoded_data, codec

        # ── Arithmetic ────────────────────────────────────────────────────────
        elif self.coding_method == 'arithmetic':
            symbols = self._blocks_to_symbols(all_frames)
            frequencies = self._build_frequencies(symbols)

            # bits=16 gives enough precision for typical coefficient distributions.
            # Raise to 20+ if you get "Insufficient precision" errors on large videos.
            encoder = ArithmeticEncoder(frequencies=frequencies, bits=16)
            bits = list(encoder.encode(iter(symbols)))

            # Return bits + everything the decoder needs to reconstruct
            metadata = {
                'frequencies': frequencies,
                'bits': 16,
                'frame_count': len(self.frames),
                'frame_shape': self.frames[0].shape,
                'block_size': self.block_size,
                'quantization_table': self.quantization_table.tolist(),
            }
            return bits, metadata

        else:
            raise ValueError(f"Unknown coding_method: {self.coding_method!r}. "f"Use 'huffman' or 'arithmetic'.")