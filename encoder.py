import cv2
import numpy as np
from scipy.fftpack import dct
import dahuffman
from arithmetic_coding import ArithmeticEncoder
from collections import Counter

class Encoder:
    def __init__(self, video_path, block_size, intra_q, inter_q, coding_method, search_range=8):
        self.video_path = video_path
        self.block_size = block_size
        self.intra_q = intra_q 
        self.inter_q = inter_q        
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
                quantized = np.round(dct_block / self.intra_q).astype(int)
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

    def process_residuals(self, residual_blocks):
        processed = []
        for block in residual_blocks:
            block_f = block.astype(np.float32)
            dct_block = dct(dct(block_f.T, norm='ortho').T, norm='ortho')
            quantized = np.round(dct_block / self.inter_q).astype(int)
            processed.append(quantized)
        return processed

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
                symbols.append(('EOB',))
        symbols.append(('<EOM>',))
        return symbols

    def motion_vectors_to_symbols(self, motion_vectors_all):
        symbols = []
        for frame_mvs in motion_vectors_all[1:]:
            for dy, dx in frame_mvs:
                symbols.append(('MV', int(dy), int(dx)))
        symbols.append(('<EOM>',))
        return symbols

    def _build_frequencies(self, symbols):
        freq = dict(Counter(symbols))
        if ('<EOM>',) not in freq:
            freq[('<EOM>',)] = 1
        return freq

    def encode(self):
        all_frames = []
        motion_vectors_all = [None]  

        for idx in range(len(self.frames)):
            if idx == 0:
                blocks = self.process_frame(self.frames[idx])
                all_frames.append(blocks)
            else:
                motion_vectors, residual_blocks = self.motion_estimate(
                    self.frames[idx], self.frames[idx - 1]
                )
                motion_vectors_all.append(motion_vectors)
                all_frames.append(self.process_residuals(residual_blocks))

        if self.coding_method == 'huffman':
            symbols = self.blocks_to_symbols(all_frames)
            huffman_codec = dahuffman.HuffmanCodec.from_data(symbols)
            encoded_data = huffman_codec.encode(symbols)

            mv_symbols = self.motion_vectors_to_symbols(motion_vectors_all)
            mv_huffman_codec = dahuffman.HuffmanCodec.from_data(mv_symbols)
            mv_encoded = mv_huffman_codec.encode(mv_symbols)
            
            return encoded_data, huffman_codec, mv_encoded, mv_huffman_codec, motion_vectors_all, self.frames, self.intra_q, self.inter_q
        
        elif self.coding_method == 'arithmetic':
            symbols = self.blocks_to_symbols(all_frames)
            frequencies = self._build_frequencies(symbols)
            arithmetic_codec = ArithmeticEncoder(frequencies=frequencies, bits=16, EOM=('<EOM>',))
            bits = list(arithmetic_codec.encode(iter(symbols)))

            mv_symbols = self.motion_vectors_to_symbols(motion_vectors_all)
            mv_frequencies = self._build_frequencies(mv_symbols)
            mv_arithmetic_codec = ArithmeticEncoder(frequencies=mv_frequencies, bits=16, EOM=('<EOM>',))
            mv_bits = list(mv_arithmetic_codec.encode(iter(mv_symbols)))

            metadata = {
                'frequencies': frequencies,
                'mv_frequencies': mv_frequencies,
                'frame_count': len(self.frames),
                'frame_shape': self.frames[0].shape,
                'block_size': self.block_size,
                'intra_quantization_table': self.intra_q.tolist(),
                'inter_quantization_table': self.inter_q.tolist(),
            }
            return bits, metadata, mv_bits, motion_vectors_all, self.frames

        else:
            raise ValueError(f"Unknown coding_method: {self.coding_method!r}. " f"Use 'huffman' or 'arithmetic'.")