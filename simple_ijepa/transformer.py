import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
import math


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim %
            4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    """As before, with an optional Dropout added AFTER the GELU and the
    output Linear (as plain submodules, NOT inside the Sequential, so the
    parameter indices -- and therefore existing checkpoints -- stay
    unchanged)."""

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.net[:-1]:
            x = layer(x)
        x = self.drop1(x)
        x = self.net[-1](x)
        return self.drop2(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # dropout on the attention probabilities and the output projection
        # (plain submodules without parameters, so existing checkpoints
        # keep loading strictly)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attn_drop(self.attend(dots))

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj_drop(self.to_out(out))


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim, heads=heads, dim_head=dim_head,
                              dropout=dropout),
                    FeedForward(dim, mlp_dim, dropout=dropout)
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class VisionTransformer(nn.Module):

    def __init__(self,
                 *,
                 image_size,
                 patch_size,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 channels=3,
                 dim_head=64,
                 dropout=0.):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                      p1=patch_height,
                      p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )
        self.pos_embedding.requires_grad_(False)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout=dropout)

    def forward(self, x, patchify=True, pos_embed=True, mask=None):
        device = x.device

        if patchify:
            x = self.to_patch_embedding(x)

        if pos_embed:
            x += self.pos_embedding.to(device, dtype=x.dtype)

        if mask is not None:
            x = torch.gather(x, 1,
                             mask.unsqueeze(-1).expand(-1, -1, x.size(-1)))

        x = self.transformer(x)
        return x
