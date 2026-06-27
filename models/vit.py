import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_embed


class SelfAttention(nn.Module):
    # Plain matmul/softmax attention, not nn.MultiheadAttention: its fused SDPA
    # kernel has no double-backward, which estimate_hessian_topk requires.
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        in_channels,
        img_size,
        num_classes=10,
        patch_size=4,
        embed_dim=64,
        depth=4,
        num_heads=4,
    ):
        super().__init__()
        self.depth = depth
        self.patch_embed = PatchEmbed(in_channels, img_size, patch_size, embed_dim)
        for i in range(depth):
            setattr(self, f"block{i}", ViTBlock(embed_dim, num_heads))
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        x = self.patch_embed(x)
        for i in range(self.depth):
            x = getattr(self, f"block{i}")(x)
        return self.head(x[:, 0])
