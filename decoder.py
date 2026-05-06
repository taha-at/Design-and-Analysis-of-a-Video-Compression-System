import numpy as np
from scipy.fftpack import idct
from arithmetic_coding import ArithmeticEncoder

class Decoder:
    def __init__(self, encoded_data, block_size, quantization_table, coding_method, frame_shape, frame_count, motion_vectors_all, codec = None, frequencies=None):
        self.encoded_data = encoded_data
        self.block_size = block_size
        self.quantization_table = np.array(quantization_table)
        self.coding_method = coding_method
        self.frequencies = frequencies
        self.frame_shape = frame_shape
        self.frame_count = frame_count
        self.motion_vectors_all = motion_vectors_all  
        self.codec = codec
        
    def entropy_decode(self):
        if self.coding_method == 'huffman':
            return self.codec.decode(self.encoded_data)

        elif self.coding_method == 'arithmetic':
            arithmetic_codec = ArithmeticEncoder(frequencies=self.frequencies, bits=16)
            return list(arithmetic_codec.decode(self.encoded_data))

        else:
            raise ValueError(f"Unsupported coding method: {self.coding_method!r}. " f"Only 'huffman' and 'arithmetic' are supported.")

    def symbols_to_blocks(self, symbols):
        blocks = []
        current = []
        n = self.block_size ** 2

        for symbol in symbols:
            if symbol == '<EOM>':
                break
            elif symbol == 'EOB':
                current.extend([0] * (n - len(current)))
                block = np.array(current[:n]).reshape(self.block_size, self.block_size)
                blocks.append(block)
                current = []
            else:
                _, run, value = symbol
                current.extend([0] * run)
                current.append(value)

        return blocks

    def inverse_dct(self, quantized_block):
        dequantized = quantized_block * self.quantization_table
        block = idct(idct(dequantized.T, norm='ortho').T, norm='ortho')
        return np.clip(block, 0, 255).astype(np.uint8)

    def reassemble_frame(self, blocks):
        h, w = self.frame_shape
        pad_h = (self.block_size - (h % self.block_size)) % self.block_size
        pad_w = (self.block_size - (w % self.block_size)) % self.block_size
        padded_h, padded_w = h + pad_h, w + pad_w

        frame = np.zeros((padded_h, padded_w), dtype=np.uint8)
        idx = 0
        for i in range(0, padded_h, self.block_size):
            for j in range(0, padded_w, self.block_size):
                frame[i:i+self.block_size, j:j+self.block_size] = blocks[idx]
                idx += 1

        return frame[:h, :w]

    def motion_compensate(self, prev_frame, motion_vectors_frame):
        h, w = self.frame_shape
        predicted = np.zeros((h, w), dtype=np.uint8)

        block_idx = 0
        for i in range(0, h - self.block_size + 1, self.block_size):  
            for j in range(0, w - self.block_size + 1, self.block_size): 
                mv_y, mv_x = motion_vectors_frame[block_idx]
                ref_y = np.clip(i + mv_y, 0, h - self.block_size)
                ref_x = np.clip(j + mv_x, 0, w - self.block_size)
                predicted[i:i+self.block_size, j:j+self.block_size] = \
                    prev_frame[ref_y:ref_y+self.block_size, ref_x:ref_x+self.block_size]
                block_idx += 1

        return predicted

    def decode(self):
        symbols  = self.entropy_decode()
        q_blocks = self.symbols_to_blocks(symbols)
        px_blocks = [self.inverse_dct(b) for b in q_blocks]

        h, w = self.frame_shape
        pad_h = (self.block_size - (h % self.block_size)) % self.block_size
        pad_w = (self.block_size - (w % self.block_size)) % self.block_size
        bpf = ((h + pad_h) // self.block_size) * ((w + pad_w) // self.block_size)

        frames = []
        for i in range(self.frame_count):
            frame_blocks = px_blocks[i * bpf : (i + 1) * bpf]
            residual = self.reassemble_frame(frame_blocks)

            if i == 0:
                frames.append(residual)  
            else:
                predicted = self.motion_compensate(frames[i - 1], self.motion_vectors_all[i])
                frame = np.clip(predicted.astype(int) + residual.astype(int), 0, 255).astype(np.uint8)
                frames.append(frame)

        return frames