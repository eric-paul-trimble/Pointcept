"""Volt: Volume Transformer backbone for point-clouds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale
from torch.nn.init import trunc_normal_

from pointcept.models.builder import MODELS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


__all__ = ["Volt"]


class Tokenizer(nn.Module):
    """Convert fine voxel features into one token per coarse voxel."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.proj = nn.Linear(kernel_size**3 * in_channels, out_channels)

    def forward(
        self, features: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return coarse features and fine-to-coarse assignment metadata."""
        kernel_size = self.kernel_size
        coarse_indices_per_voxel = indices // indices.new_tensor(
            [1, kernel_size, kernel_size, kernel_size]
        )
        coarse_indices, inverse = torch.unique(
            coarse_indices_per_voxel,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        offset = indices[:, 1:] % kernel_size
        offset_id = (
            offset[:, 0] * kernel_size * kernel_size
            + offset[:, 1] * kernel_size
            + offset[:, 2]
        )
        flat_patch_index = inverse * kernel_size**3 + offset_id
        feature_dim = features.shape[1]
        flat_patch_index = flat_patch_index.unsqueeze(1).expand(-1, feature_dim)
        patches = features.new_zeros(
            coarse_indices.shape[0] * kernel_size**3, feature_dim
        )
        # Avoid mutation through advanced indexing: scatter produces a more
        # stable graph under TorchDynamo and ONNX export.
        patches = patches.scatter(0, flat_patch_index, features)
        patches = patches.view(coarse_indices.shape[0], kernel_size**3, feature_dim)
        coarse_features = self.proj(patches.flatten(1))

        return coarse_features, coarse_indices, inverse, offset_id


class RoPE_Attention(nn.Module):
    """Multi-head self-attention with axial rotary position embeddings."""

    def __init__(
        self, dim: int = 768, num_heads: int = 12, qk_norm: bool = False
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError("dim should be divisible by num_heads")

        self.num_heads = num_heads
        self.h_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.h_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.h_dim) if qk_norm else nn.Identity()

    @staticmethod
    def apply_rotary_emb(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings using real-valued tensor operators."""
        half_head_dim = q.shape[-1] // 2
        q_even, q_odd = q.float().reshape(*q.shape[:-1], half_head_dim, 2).unbind(-1)
        k_even, k_odd = k.float().reshape(*k.shape[:-1], half_head_dim, 2).unbind(-1)
        cos = cos.to(dtype=q_even.dtype)
        sin = sin.to(dtype=q_even.dtype)
        q_rotated = torch.stack(
            (q_even * cos - q_odd * sin, q_even * sin + q_odd * cos), dim=-1
        ).flatten(-2)
        k_rotated = torch.stack(
            (k_even * cos - k_odd * sin, k_even * sin + k_odd * cos), dim=-1
        ).flatten(-2)

        return q_rotated.to(dtype=q.dtype), k_rotated.to(dtype=k.dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention to flattened tokens grouped by scene."""
        num_tokens, channels = x.shape
        qkv = self.qkv(x).view(num_tokens, 3, self.num_heads, self.h_dim)
        qkv = qkv.permute(1, 2, 0, 3)
        q, k, v = qkv.unbind(dim=0)

        q, k = self.q_norm(q).to(q.dtype), self.k_norm(k).to(k.dtype)
        q, k = self.apply_rotary_emb(q, k, cos, sin)

        # The mask is built once by Volt and reused by every transformer block.
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = attention_scores * (self.h_dim**-0.5)
        attention_scores = attention_scores.masked_fill(
            ~attention_mask.unsqueeze(0).unsqueeze(0),
            torch.finfo(attention_scores.dtype).min,
        )
        x = torch.matmul(torch.softmax(attention_scores, dim=-1), v)
        x = x.squeeze(0).transpose(0, 1).reshape(-1, channels)

        return self.proj(x)


class Block(nn.Module):
    """Transformer block composed of rotary attention and an MLP."""

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: float | None = None,
        qk_norm: bool = False,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        mlp_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = RoPE_Attention(dim, num_heads, qk_norm)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer
        )
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), cos, sin, attention_mask))
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


class RoPE(nn.Module):
    """Precompute real-valued axial rotary position embeddings."""

    def __init__(
        self,
        theta: float = 100.0,
        freq_split: tuple[int, int, int] = (12, 12, 8),
        max_grid_size: tuple[int, int, int] = (1024, 1024, 512),
        head_dim: int | None = None,
    ) -> None:
        super().__init__()

        if head_dim is not None and head_dim != 2 * sum(freq_split):
            raise ValueError(
                "RoPE frequency dimensions must contain half of the attention "
                f"head dimension; got freq_split={freq_split} and head_dim={head_dim}."
            )

        freqs_x = 1.0 / theta ** torch.linspace(0, 1, freq_split[0])
        freqs_y = 1.0 / theta ** torch.linspace(0, 1, freq_split[1])
        freqs_z = 1.0 / theta ** torch.linspace(0, 1, freq_split[2])

        for axis, freqs, max_pos in zip(
            ("x", "y", "z"), (freqs_x, freqs_y, freqs_z), max_grid_size, strict=True
        ):
            cos_cache, sin_cache = self._precompute(freqs, max_pos)
            self.register_buffer(f"cos_cache_{axis}", cos_cache, persistent=False)
            self.register_buffer(f"sin_cache_{axis}", sin_cache, persistent=False)

    @staticmethod
    def _precompute(
        freqs: torch.Tensor, max_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine and sine lookup tables for one spatial axis."""
        positions = torch.outer(torch.arange(max_pos, dtype=freqs.dtype), freqs)

        return positions.cos(), positions.sin()

    def compute_axial_cos_sin(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up and concatenate axial rotary values for voxel indices."""
        indices = indices.long()
        cos = torch.cat(
            [
                self.cos_cache_x.index_select(0, indices[:, 0]),
                self.cos_cache_y.index_select(0, indices[:, 1]),
                self.cos_cache_z.index_select(0, indices[:, 2]),
            ],
            dim=-1,
        )
        sin = torch.cat(
            [
                self.sin_cache_x.index_select(0, indices[:, 0]),
                self.sin_cache_y.index_select(0, indices[:, 1]),
                self.sin_cache_z.index_select(0, indices[:, 2]),
            ],
            dim=-1,
        )

        return cos.unsqueeze(0), sin.unsqueeze(0)


class Detokenizer(nn.Module):
    """Project coarse tokens back to fine voxel features."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.proj = nn.Linear(in_channels, kernel_size**3 * out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        coarse_features: torch.Tensor,
        inverse: torch.Tensor,
        offset_id: torch.Tensor,
    ) -> torch.Tensor:
        """Gather one projected feature vector for every fine voxel."""
        all_offsets = self.proj(coarse_features)
        all_offsets = all_offsets.view(-1, self.kernel_size**3, self.out_channels)
        flat_index = inverse * self.kernel_size**3 + offset_id
        fine_features = all_offsets.flatten(0, 1).index_select(0, flat_index)

        return fine_features + self.bias


class Decoder(nn.Module):
    """Decode coarse Volt tokens into per-point feature vectors."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        self.pre = nn.Sequential(
            nn.BatchNorm1d(in_channels, eps=1e-3, momentum=0.01),
            nn.GELU(),
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.GELU(),
        )
        self.unembed = Detokenizer(out_channels, out_channels, kernel_size)
        self.post = nn.Sequential(
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.GELU()
        )

    def forward(
        self, x: torch.Tensor, inverse: torch.Tensor, offset_id: torch.Tensor
    ) -> torch.Tensor:
        """Return decoded features in the original fine-voxel order."""
        torch._check(x.shape[0] != 0)  # Necessary for TorchDynamo and ONNX export
        x = self.pre(x)
        x = self.unembed(x, inverse, offset_id)

        return self.post(x)


@MODELS.register_module("Volt")
class Volt(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: float | None = None,
        qk_norm: bool = False,
        drop_path: float = 0.3,
        stride: int = 5,
        kernel_size: int = 5,
        up_mlp_dim: int = 256,
        increase_drop_path: bool = True,
    ) -> None:
        super().__init__()

        if stride != kernel_size:
            raise ValueError("Volt only supports non-overlapping patches")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim should be divisible by num_heads")

        self.tokenizer = Tokenizer(in_channels, embed_dim, kernel_size)

        if increase_drop_path:
            drop_path_list = torch.linspace(0, drop_path, depth).tolist()
        else:
            drop_path_list = [drop_path] * depth

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    init_values,
                    qk_norm,
                    drop_path_list[i],
                )
                for i in range(depth)
            ]
        )

        self.pos_enc = RoPE(head_dim=embed_dim // num_heads)
        self.decoder = Decoder(embed_dim, up_mlp_dim, kernel_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize linear layers with the Volt weight initialization."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "init_weights"):
            module.init_weights()

    def forward(self, data_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return per-point features for a flattened, batched point cloud."""
        grid_coord = data_dict["grid_coord"]
        feat = data_dict["feat"]
        batch = data_dict.get("batch")
        if batch is None:
            offset = data_dict["offset"]
            counts = torch.diff(torch.cat((offset.new_zeros(1), offset)))
            batch = torch.repeat_interleave(
                torch.arange(offset.shape[0], device=offset.device), counts
            )
        indices = torch.cat((batch.unsqueeze(-1), grid_coord), dim=1)
        indices = indices.to(dtype=torch.int32).contiguous()

        features, indices, inverse, offset_id = self.tokenizer(feat, indices)

        cos, sin = self.pos_enc.compute_axial_cos_sin(indices[:, 1:])
        attention_mask = indices[:, 0, None] == indices[None, :, 0]
        for block in self.blocks:
            features = block(features, cos, sin, attention_mask)
        features = self.decoder(features, inverse, offset_id)

        return features
