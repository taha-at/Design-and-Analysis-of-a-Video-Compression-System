import cv2
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import dahuffman
from arithmetic_coding import ArithmeticEncoder
from collections import Counter

class Encoder:
    def __init__(self, video_path, block_size, quantization_table, coding_method, search_range=8):
        self.video_path = video_path
        self.block_size = block_size
        self.quantization_table = quantization_table  
        self.coding_method = coding_method
        self.search_range = search_range
        
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

    def motion_estimate(self, current_frame, reference_frame):  
        h, w = current_frame.shape
        motion_vectors = []
        residual_blocks = []

        for i in range(0, h - self.block_size + 1, self.block_size):
            for j in range(0, w - self.block_size + 1, self.block_size):
                current_block = current_frame[i:i+self.block_size, j:j+self.block_size]
                best_mv, best_block = self._block_search(current_block, reference_frame, i, j, self.search_range)
                motion_vectors.append(best_mv)
                residual_blocks.append(current_block.astype(int) - best_block.astype(int))

        return motion_vectors, residual_blocks

    def _block_search(self, current_block, reference_frame, i, j, search_range):
        h, w = reference_frame.shape
        best_sad = float('inf')
        best_mv = (0, 0)
        best_block = current_block

        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                ri, rj = i + dy, j + dx
                if ri < 0 or rj < 0 or ri + self.block_size > h or rj + self.block_size > w:
                    continue
                ref_block = reference_frame[ri:ri+self.block_size, rj:rj+self.block_size]
                sad = np.sum(np.abs(current_block.astype(int) - ref_block.astype(int)))
                if sad < best_sad:
                    best_sad = sad
                    best_mv = (dy, dx)
                    best_block = ref_block

        return best_mv, best_block
    
    def blocks_to_symbols(self, all_frames):
        symbols = []
        for blocks in all_frames:
            for block in blocks:
                flat = block.flatten()  
                run = 0
                for coeff in flat:
                    if coeff == 0:
                        run += 1
                    else:
                        symbols.append(('RLE', run, int(coeff)))
                        run = 0
                symbols.append('EOB')  
        symbols.append('<EOM>')        
        return symbols

    def _build_frequencies(self, symbols):
        """Build frequency dict from symbols. ArithmeticEncoder needs this."""
        freq = dict(Counter(symbols))
        if '<EOM>' not in freq:
            freq['<EOM>'] = 1
        return freq

    def process_residuals(self, residual_blocks):
        processed = []
        for block in residual_blocks:
            block_f = block.astype(np.float32)
            dct_block = dct(dct(block_f.T, norm='ortho').T, norm='ortho')
            quantized = np.round(dct_block / self.quantization_table).astype(int)
            processed.append(quantized)
        return processed
    
    def encode(self):
        all_frames = []
        motion_vectors_all = []
        for idx in range(len(self.frames)):
            if idx == 0:
                frame = self.frames[idx]
                blocks = self.process_frame(frame)
                all_frames.append(blocks)
            else:
                current_frame = self.frames[idx]
                reference_frame = self.frames[idx - 1]
                motion_vectors, residual_blocks = self.motion_estimate(current_frame, reference_frame)
                motion_vectors_all.append(motion_vectors)
                all_frames.append(residual_blocks)
            
        if self.coding_method == 'huffman':
            flat_blocks = np.concatenate(
                [np.array(b).flatten() for b in all_frames]
            ).flatten()
            codec = dahuffman.HuffmanCodec.from_data(flat_blocks)
            encoded_data = codec.encode(flat_blocks)
            return encoded_data, codec, motion_vectors_all

        elif self.coding_method == 'arithmetic':
            symbols = self.blocks_to_symbols(all_frames)
            frequencies = self._build_frequencies(symbols)
            encoder = ArithmeticEncoder(frequencies=frequencies, bits=16)
            bits = list(encoder.encode(iter(symbols)))
            metadata = {
                'frequencies': frequencies,
                'bits': 16,
                'frame_count': len(self.frames),
                'frame_shape': self.frames[0].shape,
                'block_size': self.block_size,
                'quantization_table': self.quantization_table.tolist(),
            }
            return bits, metadata, motion_vectors_all

        else:
            raise ValueError(f"Unknown coding_method: {self.coding_method!r}. "f"Use 'huffman' or 'arithmetic'.")