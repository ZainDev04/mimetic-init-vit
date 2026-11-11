# Vanilla Vision Transformer (ViT) for CIFAR-10, written in plain PyTorch.
# Default size follows the paper's main CIFAR-10 model: width 192, depth 12, 3 heads, patch size 2.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_position_embedding(num_tokens, dim):
    # Fixed (not learned) position embeddings from "Attention Is All You Need".
    # Row i is the code for position i. Even columns use sin, odd columns use cos,
    # each pair at a different frequency, so nearby positions get similar codes.
    position = torch.arange(num_tokens).unsqueeze(1)                          # (num_tokens, 1)
    freq = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))   # (dim/2,)
    pe = torch.zeros(num_tokens, dim)
    pe[:, 0::2] = torch.sin(position * freq)
    pe[:, 1::2] = torch.cos(position * freq)
    return pe                                                                 # (num_tokens, dim)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        # Separate layers for query, key and value so mimetic init can set each one.
        # nn.Linear stores weight as (out, in) and computes x @ weight.T + bias,
        # so the paper's W_Q (dim x dim, used as x @ W_Q) is q.weight.T here.
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)   # the paper's W_proj: mixes the heads back together

    def forward(self, x):
        B, N, D = x.shape                 # batch, tokens, embedding dim
        # Project, then split the last dim into heads: (B, N, D) -> (B, heads, N, head_dim)
        q = self.q(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        # Attention: out = softmax(q @ k^T / sqrt(head_dim)) @ v, giving (B, heads, N, head_dim).
        # Each row of the softmax says how much one token looks at every other token.
        # PyTorch's built-in function computes exactly this formula, but in blocks, without
        # storing the full (B, heads, N, N) matrix. Written out by hand, that matrix needs
        # about 0.6 GB per layer at batch 512 and the T4 runs out of memory.
        out = F.scaled_dot_product_attention(q, k, v)
        # Join the heads again: (B, heads, N, head_dim) -> (B, N, D)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class Block(nn.Module):
    # One transformer layer, pre-norm: LayerNorm comes before attention and before the MLP.
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # residual connection around attention
        x = x + self.mlp(self.norm2(x))   # residual connection around the MLP
        return x


class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=2, dim=192, depth=12, heads=3, num_classes=10, pos_scale=1.0):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2      # 32/2 = 16, so 16 * 16 = 256 patches
        # Cut the image into patch_size x patch_size squares and map each to a dim-vector.
        # A conv with kernel = stride = patch_size does exactly this in one step.
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        # Extra learned token; its final state is used to classify the image.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # Fixed sinusoidal position embeddings for the class token + all patches.
        # A buffer is saved with the model and moved to the GPU, but not trained.
        # pos_scale is gamma in Fig. 4 of the paper: a larger value makes position matter more.
        self.register_buffer("pos_embed", pos_scale * sinusoidal_position_embedding(num_patches + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.patch_embed(img)              # (B, 3, 32, 32) -> (B, dim, 16, 16)
        x = x.flatten(2).transpose(1, 2)       # (B, dim, 256) -> (B, 256, dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)   # one copy per image: (B, 1, dim)
        x = torch.cat([cls, x], dim=1)         # (B, 257, dim)
        x = x + self.pos_embed                 # add position codes
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])              # classify from the class token: (B, num_classes)


if __name__ == "__main__":
    # Quick check: run on Colab with `python model.py`.
    model = ViT()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.2f}M")
    out = model(torch.randn(2, 3, 32, 32))
    print("output shape:", tuple(out.shape))  # expect (2, 10)
    # Check that the built-in attention equals the formula written out by hand.
    q, k, v = torch.randn(3, 2, 3, 257, 64)   # (batch, heads, tokens, head_dim) each
    manual = (q @ k.transpose(-2, -1) / math.sqrt(64)).softmax(dim=-1) @ v
    diff = (F.scaled_dot_product_attention(q, k, v) - manual).abs().max().item()
    print(f"max difference to hand-written attention: {diff:.1e}")  # expect about 1e-6 or less
    # Check pos_scale: the position embeddings should be exactly twice as large.
    ratio = (ViT(pos_scale=2.0).pos_embed / model.pos_embed)[1:]   # skip row 0: sin(0) = 0 gives 0/0
    print("pos_scale 2 ratio:", ratio.min().item(), ratio.max().item())  # expect 2.0 2.0
