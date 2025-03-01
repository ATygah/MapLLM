import torch
import torch.nn as nn
#import torch.nn.functional as F
from fc import calculate_ff_costs
from attn import generic_qk_attention_flops,generic_attention_value_flops,softmax_flops

def my_attention(query, key, value, num_heads=1, return_attn=False):
    # query: a tensor of [batch, len_query, dim]
    # key: a tensor of [batch, len_key, dim]
    # value: a tensor of [batch, len_value, dim]
    batch, len_query, dim = query.size()
    len_key = key.size(-2)
    len_value = value.size(-2)

    ###### YOUR CODE BEGINS HERE ######
    # which two of `len_query`, `len_key` and `len_value` always equal in both self-attention and cross-attention?
    # modify the assertion conditional below

    # Since len_key and len_value are generated from the same source, they should be equal in length.
    # A simple analogy is that of a database. The query is indepedent from the database. The key and value are tied to each
    # other. Each key corresponds to a value.
    assert(len_key == len_value)
    ######  YOUR CODE ENDS HERE  ######

    head_dim = dim//num_heads

    ###### YOUR CODE BEGINS HERE ######
    # implement attention mechanism here
    query = query.view(batch,len_query,num_heads, head_dim)
    key = key.view(batch,len_key,num_heads, head_dim)
    value = value.view(batch,len_value,num_heads, head_dim)
    # Now we have split up the tensors.

    #Transposing to get all queries within on head together.
    query = query.transpose(1,2)
    key = key.transpose(1,2)
    value = value.transpose(1,2)

    atten_score = query@key.transpose(2,3)

    attn_weights = torch.softmax(atten_score/key.shape[-1]**0.5, dim=-1)

    attn_output = attn_weights@value
    attn_output = attn_output.transpose(1,2)
    attn_output = attn_output.contiguous().view(batch,len_query,dim)

    qk_flops = generic_qk_attention_flops(batch, num_heads, len_query, len_key, head_dim)
    av_flops = generic_attention_value_flops(batch, num_heads, len_query, len_key, head_dim)
    sftmax_flops = softmax_flops(batch, len_query, len_key, num_heads)
    print(f"QK FLOPS: {qk_flops / 1e9:.2f} GFLOPS, AV FLOPS: {av_flops / 1e9:.2f} GFLOPS, SFTMAX FLOPS: {sftmax_flops / 1e9:.2f} GFLOPS")
    flops = qk_flops + av_flops + sftmax_flops

    ######  YOUR CODE ENDS HERE  ######
    if return_attn:
        return attn_output, attn_weights, flops
    else:
        return attn_output, flops

class MySelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.in_proj = nn.Linear(dim, 3*dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dim = dim
        self.num_heads = num_heads
        self.attn_weights = None
        self.flops = 0  # Track FLOPs for this block

    def forward(self, x):
        batch, seq_len, dim = x.shape

        # Calculate FLOPs for in_proj (input projection)
        in_flops, _, _ = calculate_ff_costs(
            batch_size=batch,
            seq_len=seq_len,
            embed_dim=dim,
            vocab_size=3*dim  # in_proj expands to 3*dim
        )
        
        qkv = self.in_proj(x)  # Shape: (batch, seq_len, 3*dim)
        query, key, value = qkv.chunk(3, dim=-1)  # Split last dim into 3 parts

        out_attn, attn_weights, attn_flops = my_attention(query, key, value, self.num_heads, return_attn=True)
        self.attn_weights = attn_weights.detach().cpu()

        # Calculate FLOPs for out_proj (output projection)
        out_flops, _, _ = calculate_ff_costs(
            batch_size=batch,
            seq_len=seq_len,
            embed_dim=dim,
            vocab_size=dim  # out_proj maps back to dim
        )
        
        out = self.out_proj(out_attn)
        
        # Total FLOPs for this block
        self.flops = in_flops + out_flops + attn_flops
        
        return out

    def get_flops(self):
        return self.flops

# Create attention block
attn_block = MySelfAttentionBlock(dim=768, num_heads=12)

# Forward pass with dummy input
x = torch.randn(8, 1024, 768)  # (batch, seq_len, dim)
output = attn_block(x)

# Get FLOPs
print(f"Total FLOPs for attention block: {attn_block.get_flops() / 1e9:.2f} GFLOPs")

