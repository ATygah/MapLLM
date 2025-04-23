# This file is used to map the PEs to the attention layer.
# It uses the NoCResources class to get the PEs required for each layer and block.
# It then uses the PEMap class to map the PEs to the attention layer.

from attn import *
import math
import numpy as np
import matplotlib.pyplot as plt
import time

'''
model_name: "gpt2-small"
attention_type: "self"
vocab_size: 50257
embed_dim: 768
max_seq_len: 1024
batch_size: 1
seq_len: 1024
kv_seq_len: 1024
dtype: "float32"
num_heads: 12
num_layers: 12
mlp_expansion_factor: 4
mlp_activation: "gelu"
norm_type: "layernorm"
dropout_rate: 0.1
architecture: "gpt2" 
'''
class PEMap:
    def __init__(self, pes_required= None):
        self.pes_required = pes_required

    def map_pes_to_attention(self):

        # Hardcoding, remove later.
        memory_per_pe = 64 * 1024 # 64KB
        dtype_bytes = dtype_size('float16')
        num_heads = 12
        d_model = 768
        d_out = 768
        seq_len = 1024
        batch_size = 1

        '''
        ---- MATH LEVEL IMPLEMENTATION ----
        input_bytes = seq_len * d_model * dtype_bytes
        print(f"Input bytes: {input_bytes/1024} KB")

        # Wq, Wk, Wv are 768x768 dimensional.
        # Wq_head, Wk_head, Wv_head are 768x64 dimensional.
        Wq_params, Wq_bytes = weight_matrix_parameters(d_model=768, d_out=768, include_bias=False, dtype='float16')
        Wq_bytes_head = Wq_bytes / num_heads
        print(f"Wq bytes: {Wq_bytes/1024} KB")
        print(f"Wq bytes per head: {Wq_bytes_head/1024} KB")
        
        # Output of inp x Wq is Q 1024x768.
        # Q_head 1024x64.
        Q_bytes = seq_len * d_out * dtype_bytes
        Q_bytes_head = seq_len * (d_out // num_heads) * dtype_bytes
        print(f"Q bytes: {Q_bytes/1024} KB")
        print(f"Q bytes per head: {Q_bytes_head/1024} KB")
        '''

        '''
        ---- TENSOR LEVEL IMPLEMENTATION ----
        '''
        dtype = torch.float16
        d_head = d_out // num_heads
        
        #input matrix
        input_matrix = torch.randn(seq_len, d_model, dtype=dtype)
        input_bytes = input_matrix.element_size() * input_matrix.nelement()
        print(f"Input size: {input_matrix.size()}")
        print(f"Input bytes: {input_bytes/1024} KB")

        # Create Wq matrix        
        Wq = torch.randn(d_model, d_out, dtype=dtype)
        Wq_bytes = Wq.element_size() * Wq.nelement()
        print(f"Wq size: {Wq.size()}")
        print(f"Wq bytes: {Wq_bytes / 1024} KB")

        # Create Q matrix
        Q = torch.matmul(input_matrix, Wq)
        print(f"Q size: {Q.size()}")
        Q_bytes = Q.element_size() * Q.nelement()
        print(f"Q bytes: {Q_bytes / 1024} KB")

        # Create Q per head
        # Q will be split into multiple heads along the d_out dimension.
        Wq_heads = torch.chunk(Wq, num_heads, dim=1)        # List of 12 tensors, each of shape (768, 64)
        Q_heads = torch.chunk(Q, num_heads, dim=1)        # List of 12 tensors, each of shape (1024, 64)

        # Verify that the total bytes of Q_heads match the original Q bytes
        total_Q_heads_bytes = sum(qh.element_size() * qh.nelement() for qh in Q_heads)
        print(f"Total Q_heads bytes: {total_Q_heads_bytes / 1024} KB")  # Should match Q_bytes / 1024

        # Iterate through each Q_head and print its size and memory usage
        print(f"Number of elements in Q_heads: {len(Q_heads)}")
        Q_head = Q_heads[0]
        print(f"Dimension of Q_head: {Q_head.size()}")
        Q_bytes_head_individual = Q_head.element_size() * Q_head.nelement()
        print(f"Q_head size: {Q_head.size()}")  # Expected: torch.Size([1024, 64])
        print(f"Q_head bytes: {Q_bytes_head_individual / 1024} KB")  # Expected: 128.0 KB per head
        print("------------------------------------------------------------------------------")

        print("Matrix division per head")
        # Defining three variables which will tell how many times we divide a matrix.
        # Assuming that inputs will be divided into 4 parts.
        # And one row will get the one of the 4 parts.
        seq_split = 4
        # Assuming that the d_model will be divided into 12 parts.
        # And we will send one of the 12 parts at a time and keep sendin the other parts in the next cycle.
        # This will be done for all the 12 parts.
        # And those parts will keep getting added up and store in the Q(256x16) matrix.
        d_model_split = 12
        # Assuming that the d_head will be divided into 4 parts.
        # And one row will get the one of the 4 parts.
        d_head_split = 4

        

def main():
    # Initialize the PEMAP with a sample required PE count (adjust as needed)
    #pes_required = 64  # Example value; modify based on your requirements
    pe_map = PEMap()
    pe_map.map_pes_to_attention()

if __name__ == "__main__":
    main()






