import torch
from torch.utils.flop_counter import FlopCounterMode
from attn import multihead_self_attention_flops, cross_attention_flops

# Self-attention test case
d_in = 512
d_out = 512
batch_size = 2
seq_length = 100
num_heads = 8

# Theoretical calculation
theoretical_self_attn_flops = multihead_self_attention_flops(
    d_in, d_out, batch_size, seq_length, num_heads
)

# PyTorch model
model_self_attn = torch.nn.MultiheadAttention(
    embed_dim=d_out,
    num_heads=num_heads,
    bias=False,
    batch_first=False
).eval()

# Input tensor (seq_length, batch_size, embed_dim)
x = torch.randn(seq_length, batch_size, d_in)

# FLOP measurement
with FlopCounterMode() as flop_counter:
    model_self_attn(x, x, x)
measured_self_attn_flops = flop_counter.get_total_flops()

# Cross-attention test case
d_in_q = 256
d_in_kv = 512
q_seq_length = 50
kv_seq_length = 100

# Theoretical calculation
theoretical_cross_attn_flops = cross_attention_flops(
    d_in_q, d_in_kv, d_out, batch_size, q_seq_length, kv_seq_length, num_heads
)

# PyTorch model
class CrossAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(d_in_q, d_out, bias=False)
        self.k_proj = torch.nn.Linear(d_in_kv, d_out, bias=False)
        self.v_proj = torch.nn.Linear(d_in_kv, d_out, bias=False)
        self.out_proj = torch.nn.Linear(d_out, d_out, bias=False)
        self.num_heads = num_heads

    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        batch_size = query.size(1)
        q = q.contiguous().view(q.size(0), batch_size * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(k.size(0), batch_size * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(v.size(0), batch_size * self.num_heads, -1).transpose(0, 1)
        
        # Attention computation
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )
        
        # Reshape back and output projection
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, batch_size, d_out)
        return self.out_proj(attn_output)

model_cross_attn = CrossAttention().eval()
query = torch.randn(q_seq_length, batch_size, d_in_q)
key = value = torch.randn(kv_seq_length, batch_size, d_in_kv)

# FLOP measurement
with FlopCounterMode() as flop_counter:
    model_cross_attn(query, key, value)
measured_cross_attn_flops = flop_counter.get_total_flops()

# Results comparison
print(f"Self-attention FLOPs:")
print(f"Theoretical: {theoretical_self_attn_flops}")
print(f"Measured:    {measured_self_attn_flops}")
print(f"Difference:  {abs(theoretical_self_attn_flops - measured_self_attn_flops) / theoretical_self_attn_flops:.2%}\n")

print(f"Cross-attention FLOPs:")
print(f"Theoretical: {theoretical_cross_attn_flops}")
print(f"Measured:    {measured_cross_attn_flops}")
print(f"Difference:  {abs(theoretical_cross_attn_flops - measured_cross_attn_flops) / theoretical_cross_attn_flops:.2%}") 