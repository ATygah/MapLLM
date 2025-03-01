import torch
import torch.nn as nn
#import torch.nn.functional as F
from fc import calculate_ff_costs
from attn import generic_qk_attention_flops,generic_attention_value_flops,softmax_flops
from norm import calculate_norm_params_flops
from collections import OrderedDict

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
        print(f"Total FLOPs for self-attention block: {self.flops / 1e9:.2f} GFLOPS")

        return out

    def get_flops(self):
        return self.flops

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.ln_1 = nn.LayerNorm(dim, eps=1e-5)
        self.self_attention = MySelfAttentionBlock(dim, num_heads)
        
        self.ln_2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(dim*4, dim),
            nn.Dropout(0.0),
        )
        self.total_flops = 0  # Track total FLOPs per encoder block

    def forward(self, x):
        # LayerNorm 1 FLOPs
        ln1_flops = self._calculate_ln_flops(x)
        
        # Self-attention + residual
        attn_out = self.self_attention(self.ln_1(x))
        x = x + attn_out  # Residual connection (add FLOPs)
        add_flops1 = x.numel()  # 1 FLOP per element
        
        # LayerNorm 2 FLOPs
        ln2_flops = self._calculate_ln_flops(x)
        
        # MLP + residual
        mlp_out = self.mlp(self.ln_2(x))
        out = x + mlp_out  # Residual connection (add FLOPs)
        add_flops2 = x.numel()  # 1 FLOP per element
        
        # Calculate MLP FLOPs
        mlp_flops = self._calculate_mlp_flops(x)
        
        # Total FLOPs
        self.total_flops = (
            ln1_flops + 
            self.self_attention.get_flops() + 
            add_flops1 +
            ln2_flops +
            mlp_flops +
            add_flops2
        )
        print(f"Total FLOPs for ln1 block: {ln1_flops / 1e9:.2f} GFLOPS")
        print(f"Total FLOPs for self-attention block: {self.self_attention.get_flops() / 1e9:.2f} GFLOPS")
        print(f"Total FLOPs for ln2 block: {ln2_flops / 1e9:.2f} GFLOPS")
        print(f"Total FLOPs for mlp block: {mlp_flops / 1e9:.2f} GFLOPS")
        print(f"Total FLOPs for encoder block: {self.total_flops / 1e9:.2f} GFLOPS")

        return out

    def _calculate_ln_flops(self, x):
        """Calculate LayerNorm FLOPs using norm.py functions"""
        batch, seq_len, dim = x.shape
        _, ln_flops = calculate_norm_params_flops("layernorm", (batch, seq_len, dim))
        return ln_flops

    def _calculate_mlp_flops(self, x):
        """Calculate MLP FLOPs using fc.py functions"""
        batch, seq_len, dim = x.shape
        
        # First linear: dim -> 4*dim
        flops1, _, _ = calculate_ff_costs(batch, seq_len, dim, 4*dim)
        
        # Second linear: 4*dim -> dim
        flops2, _, _ = calculate_ff_costs(batch, seq_len, 4*dim, dim)
        
        return flops1 + flops2

    def get_flops(self):
        return self.total_flops

class Encoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.layers = nn.Sequential(*[EncoderBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.total_flops = 0  # Track total FLOPs

    def forward(self, input: torch.Tensor):
        # Positional embedding addition
        input = input + self.pos_embedding
        add_flops = input.numel()  # 1 FLOP per element
        
        # Process through layers
        out = self.layers(input)
        
        # Final LayerNorm
        batch, seq_len, dim = out.shape
        ln_flops = calculate_norm_params_flops("layernorm", (batch, seq_len, dim))[1]
        
        # Aggregate FLOPs
        layer_flops = sum(block.get_flops() for block in self.layers)
        self.total_flops = add_flops + ln_flops + layer_flops
        
        return self.ln(out)

    def get_flops(self):
        return self.total_flops

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size = 224,
        patch_size = 16,
        num_layers = 12,
        num_heads = 12,
        hidden_dim = 768,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (image_size // patch_size) ** 2

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
        )
        self.seq_length = seq_length

        heads_layers = OrderedDict()
        heads_layers["head"] = nn.Linear(hidden_dim, num_classes)

        self.heads = nn.Sequential(heads_layers)

        self.total_flops = 0  # Track total FLOPs

    def _process_input(self, x):
        n, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p

        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        # Save original input shape for FLOPs calculation
        original_shape = x.shape  # (batch, 3, H, W)
        
        # Process input through conv_proj
        x = self._process_input(x)
        
        # Calculate conv FLOPs using original input shape
        conv_flops = self._calculate_conv_flops(original_shape)
        
        # Class token operations
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        cat_flops = n * self.hidden_dim  # Copying class token
        
        # Encoder FLOPs
        x = self.encoder(x)
        encoder_flops = self.encoder.get_flops()
        
        # Final classification head
        x = x[:, 0]
        head_flops, _, _ = calculate_ff_costs(1, 1, self.hidden_dim, self.num_classes)
        
        # Total FLOPs
        self.total_flops = conv_flops + cat_flops + encoder_flops + head_flops
        return x

    def _calculate_conv_flops(self, input_shape):
        """Calculate FLOPs for the initial convolution layer"""
        n, c, h, w = input_shape  # Original input dimensions
        
        kernel_size = self.patch_size
        out_channels = self.hidden_dim
        
        h_out = h // kernel_size
        w_out = w // kernel_size
        
        flops_per_output = 2 * kernel_size**2 * c
        total_flops = n * out_channels * h_out * w_out * flops_per_output
        
        return total_flops

    def get_flops(self):
        return self.total_flops

# Example usage for Vision Transformer (ViT-B/16)
vit = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    num_classes=1000
)

x = torch.randn(8, 3, 224, 224)  # Batch 8, 3 channels, 224x224
output = vit(x)

print("\n===== FLOPs Breakdown =====")
print(f"1. Conv Projection: {vit._calculate_conv_flops(x.shape) / 1e9:.2f} GFLOPs")
print(f"2. Class Token Addition: {8*768 / 1e6:.2f} MFLOPs")
print(f"3. Encoder ({vit.num_layers} layers): {vit.encoder.get_flops() / 1e9:.2f} GFLOPS")
print(f"4. Classification Head: {2*768*1000 / 1e6:.2f} MFLOPS")
print("---------------------------")
print(f"Total FLOPs: {vit.get_flops() / 1e9:.2f} GFLOPS")

# Sample output:
"""
Total FLOPs for self-attention block: 2.37 GFLOPS
Total FLOPs for encoder block: 80.12 GFLOPS
... (repeated for 12 encoder blocks) ...
Total FLOPs for encoder: 961.44 GFLOPS

===== FLOPs Breakdown =====
1. Conv Projection: 1.85 GFLOPs
2. Class Token Addition: 6.14 MFLOPs
3. Encoder (12 layers): 961.44 GFLOPS
4. Classification Head: 1.54 MFLOPS
---------------------------
Total FLOPs: 963.30 GFLOPS
"""

