"""LinNet point-cloud backbone."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyg_lib.ops as pyg_ops
import torch
import torch.nn as nn
from linnetops_cuda import insert_hashtable_cuda, linear_voxel_query_cuda
from pointops._C import knn_query_cuda
from torch.autograd import Function

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, offset2bincount

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

__all__ = ["LinNet"]


def max_pool(features: torch.Tensor) -> torch.Tensor:
    """Reduce the neighbourhood dimension with an export-friendly operator."""
    return torch.amax(features, dim=-2)


def xyz2grid_id(
    xyz: torch.Tensor, grid_size: float, offset: torch.Tensor
) -> torch.Tensor:
    """Compute a packed 64-bit voxel-grid ID for each point.

    The layout is ``[batch:8 | z:18 | y:18 | x:18]``.  Batch IDs must be
    below 256 and each coordinate must fit in 18 bits.
    """
    batch = offset2batch(offset)
    coords = torch.floor_divide(xyz, grid_size).to(torch.long)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    key = (
        (x & ((1 << 18) - 1))
        | ((y & ((1 << 18) - 1)) << 18)
        | ((z & ((1 << 18) - 1)) << 36)
        | ((batch & ((1 << 8) - 1)) << 54)
    )

    return key


def counts2pos(counts: torch.Tensor) -> torch.Tensor:
    """Pack sorted voxel start positions and counts into 64-bit values."""
    value = torch.zeros_like(counts, dtype=torch.int64)
    start = torch.cumsum(counts, dim=0)
    start = torch.cat([start.new_zeros(1), start[:-1]])
    value |= (start & ((1 << 32) - 1)) << 0
    value |= (counts & ((1 << 32) - 1)) << 32

    return value


def next_power_of_2(n: int) -> int:
    """Return the smallest power of two greater than or equal to ``n``."""
    if n < 1:
        return 1

    return 1 << (n - 1).bit_length()


def xyz_height(xyz: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Return each point's height above the minimum Z in its scene."""
    z = xyz[:, 2]
    idx_ptr = torch.cat([offset.new_zeros(1), offset]).long()
    lowest_z, _ = pyg_ops.segment_min_csr(z, idx_ptr)
    lowest_z = lowest_z.repeat_interleave(offset2bincount(offset))

    return z.unsqueeze(-1) - lowest_z.unsqueeze(-1)


def _inverse_distance_interpolate(
    features: torch.Tensor, distances: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Interpolate neighbourhood features using inverse-distance weights."""
    exact = distances <= eps
    inverse_distance = torch.reciprocal(distances.clamp_min(eps))
    inverse_weights = inverse_distance / inverse_distance.sum(dim=-1, keepdim=True)
    exact_weights = exact.to(inverse_distance.dtype) / exact.sum(
        dim=-1, keepdim=True
    ).clamp_min(1)
    weights = torch.where(
        exact.any(dim=-1, keepdim=True), exact_weights, inverse_weights
    )

    return torch.sum(features * weights.unsqueeze(-1), dim=-2)


class PointBatchNorm(nn.Module):
    """Apply ``BatchNorm1d`` to flattened or neighbourhood point features."""

    def __init__(self, embed_channels: int) -> None:
        super().__init__()

        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            return (
                self.norm(features.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )

        if features.dim() == 2:
            return self.norm(features)

        raise ValueError("PointBatchNorm expects a 2D or 3D tensor")


def create_linear_block1d(
    in_channels: int, out_channels: int, bn: bool = True, act: bool = True
) -> nn.Sequential:
    """Build a ``Linear -> [BatchNorm] -> [ReLU]`` block."""
    layers: list[nn.Module] = [nn.Linear(in_channels, out_channels)]
    if bn:
        layers.append(PointBatchNorm(out_channels))
    if act:
        layers.append(nn.ReLU(inplace=False))

    return nn.Sequential(*layers)


class KNNQuery(Function):
    """Run LinNet's CUDA batched KNN query."""

    @staticmethod
    def forward(
        ctx: Any,
        nsample: int,
        xyz: torch.Tensor,
        offset: torch.Tensor,
        new_xyz: torch.Tensor | None = None,
        new_offset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return KNN indices and distances for one or more point clouds."""
        if new_xyz is None or new_offset is None:
            new_xyz = xyz
            new_offset = offset

        assert xyz.is_contiguous() and new_xyz.is_contiguous()

        m = new_xyz.shape[0]
        idx = torch.zeros((m, nsample), dtype=torch.int32, device=xyz.device)
        dist2 = torch.zeros((m, nsample), dtype=torch.float32, device=xyz.device)
        knn_query_cuda(
            m, nsample, xyz, new_xyz, offset.int(), new_offset.int(), idx, dist2
        )

        return idx, torch.sqrt(dist2)


class HashQuery(Function):
    """Find nearby points with LinNet's CUDA hash-grid implementation."""

    @staticmethod
    def forward(
        ctx: Any,
        nsample: int,
        xyz: torch.Tensor,
        offset: torch.Tensor,
        new_xyz: torch.Tensor,
        new_offset: torch.Tensor,
        grid_size: float = 0.08,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hash-grid neighbour indices and distances."""
        assert xyz.is_contiguous()

        xyz = xyz - torch.min(xyz, dim=0)[0].unsqueeze(0)
        new_xyz = new_xyz - torch.min(new_xyz, dim=0)[0].unsqueeze(0)
        grid_id = xyz2grid_id(xyz, grid_size, offset)
        new_grid_id = xyz2grid_id(new_xyz, grid_size, new_offset)

        key, counts = torch.unique(grid_id, sorted=True, return_counts=True)
        index = torch.argsort(grid_id)
        sorted_xyz = xyz.view(-1, 3)[index, :]

        value = counts2pos(counts)
        num_keys = key.numel()
        hash_size = next_power_of_2(num_keys * 2)

        hash_table = torch.zeros(hash_size * 2, dtype=torch.long, device=xyz.device)
        insert_hashtable_cuda(num_keys, hash_size, hash_table, key, value)

        new_xyz = new_xyz.view(-1, 3)

        n = new_xyz.size(0)
        idx = torch.zeros((n, nsample), dtype=torch.int32, device=xyz.device)
        dist = torch.zeros((n, nsample), dtype=torch.float32, device=xyz.device)

        linear_voxel_query_cuda(
            n,
            nsample,
            hash_size,
            grid_size,
            hash_table,
            sorted_xyz,
            new_xyz,
            new_grid_id,
            new_offset.int(),
            idx,
            dist,
        )

        return index[idx.long()], dist


knn_query = KNNQuery.apply
hash_query = HashQuery.apply


class DownSample(nn.Module):
    """Downsample features with PointNet++-style local positional encoding."""

    def __init__(
        self, in_channels: int, out_channels: int, nsample: int, radius: float
    ) -> None:
        super().__init__()

        self.nsample = nsample
        self.radius = radius

        self.conv1 = create_linear_block1d(in_channels, out_channels)
        # for positional encoding on deltas
        self.conv2 = nn.Sequential(
            create_linear_block1d(3, 32),
            create_linear_block1d(32, 32),
            create_linear_block1d(32, out_channels, act=False),
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather coarse points and pool features from their KNNs."""
        p, f, index, offset = inputs
        new_p = p[index, :]
        f = self.conv1(f)

        knn_idx, _ = knn_query(
            self.nsample, p, offset[0].contiguous(), new_p, offset[1].contiguous()
        )
        knn_idx = knn_idx.long()

        dp = p[knn_idx] - new_p.unsqueeze(1)
        pe = self.conv2(dp)

        f = pe + f[knn_idx]
        f = self.bn(_max_pool(f))

        return new_p, f


class DSA(nn.Module):
    """Dynamic spatial aggregation block used inside an encoder stage."""

    def __init__(self, in_channels: int, out_channels: int, expanse: int = 1) -> None:
        super().__init__()

        self.pre = create_linear_block1d(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

        # feed-forward network
        c = [out_channels, out_channels * expanse, out_channels]
        layers = [
            create_linear_block1d(c[i], c[i + 1], act=(i < len(c) - 2))
            for i in range(len(c) - 1)
        ]
        self.ffn = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=False)

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate positional and feature information for one point scale."""
        p, f, pe, knn_index = inputs
        identity = f
        f = self.pre(f)
        f = self.bn(_max_pool(pe + f[knn_index]))

        f = self.ffn(f)

        f = identity + f
        f = self.act(f)

        return p, f, pe, knn_index


class Stage(nn.Module):
    """One LinNet encoder stage: downsample, then aggregate locally."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        nsample: int,
        radius: Sequence[float],
    ) -> None:
        super().__init__()
        if len(radius) != 2:
            raise ValueError("each stage requires two radius values")
        self.radius = tuple(radius)
        self.down_sample = DownSample(in_channels, out_channels, nsample, radius[0])
        self.blocks = nn.Sequential(
            *(DSA(out_channels, out_channels) for _ in range(depth - 1))
        )
        # PE encoder
        self.pe = nn.Sequential(
            create_linear_block1d(3, 32),
            create_linear_block1d(32, 32),
            create_linear_block1d(32, out_channels, act=False),
        )

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one point scale and return its coordinates and features."""
        p, f, index, offset = inputs

        new_p, new_f = self.down_sample((p, f, index, offset))

        knn_idx, _ = hash_query(
            self.down_sample.nsample,
            new_p,
            offset[1].contiguous(),
            new_p,
            offset[1].contiguous(),
            self.radius[1] * 4,
        )
        knn_idx = knn_idx.long().contiguous()
        dp = new_p[knn_idx] - new_p.unsqueeze(1)
        pe = self.pe(dp)

        for block in self.blocks:
            new_p, new_f, pe, knn_idx = block((new_p, new_f, pe, knn_idx))
        return new_p, new_f


class FeaturePropogation(nn.Module):
    """Propagate coarse features back to a finer point scale."""

    def __init__(
        self, mlp: Sequence[int], nsample: int, interpolate: bool = True
    ) -> None:
        super().__init__()

        self.f2conv = create_linear_block1d(mlp[1], mlp[0])
        self.norm = PointBatchNorm(mlp[0])
        self.nsample = nsample
        self.interpolate = interpolate

    def forward(
        self,
        pfo1: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pfo2: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Propagate features from ``pfo2`` onto the points in ``pfo1``."""
        p1, f1, o1 = pfo1
        p2, f2, o2 = pfo2

        idx, distances = knn_query(
            self.nsample, p2, o2.contiguous(), p1, o1.contiguous()
        )
        propagated = self.f2conv(f2[idx.long(), :])
        if self.interpolate:
            propagated = _inverse_distance_interpolate(propagated, distances)
        else:
            propagated = _max_pool(propagated)
        f1 = f1 + propagated
        f1 = self.norm(f1)

        return f1


@MODELS.register_module()
class LinNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        nsample: int = 24,
        grid_sizes: Sequence[float] = (0.15, 0.375, 0.9375, 2.34375),
        channels: Sequence[int] = (64, 128, 256, 512, 1024),
        encoder_depth: Sequence[int] = (4, 4, 7, 4),
        radius: Sequence[float] = (0.05, 0.15, 0.375, 0.9375, 2.34375),
        decoder_nsample: int = 1,
        use_height: bool = False,
        interpolate: bool = False,
    ) -> None:
        super().__init__()

        if len(channels) != len(encoder_depth) + 1:
            raise ValueError("channels must contain one more value than encoder_depth")
        if len(radius) != len(encoder_depth) + 1:
            raise ValueError("radius must contain one more value than encoder_depth")
        self.grid_sizes = tuple(grid_sizes)
        self.use_height = use_height
        self.interpolate = interpolate
        self.emb = create_linear_block1d(in_channels + int(use_height), channels[0])
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(len(encoder_depth)):
            enc = Stage(
                channels[i],
                channels[i + 1],
                depth=encoder_depth[i],
                nsample=nsample,
                radius=radius[i : i + 2],
            )
            self.enc_stages.append(enc)
            dec = FeaturePropogation(
                channels[i : i + 2], nsample=decoder_nsample, interpolate=interpolate
            )
            self.dec_stages.append(dec)

    def forward(self, data_dict: Mapping[str, Any]) -> torch.Tensor:
        """Encode the prepared point batch and return per-point features."""
        if "offset1" not in data_dict or "indices" not in data_dict:
            raise ValueError(
                "LinNet requires engine-side multiscale preparation via "
                "prepare_linnet_input"
            )

        p0 = data_dict["coord"]
        f0 = data_dict["feat"]
        offset = data_dict["offset1"]
        indices = data_dict["indices"]
        p, f = [p0], [f0]
        if self.use_height:
            f0 = torch.cat((f0, xyz_height(p0, offset[0])), dim=-1)
        _f = self.emb(f0)
        _p = p0
        f.append(_f)
        p.append(_p)
        for i in range(len(self.enc_stages)):
            _p, _f = self.enc_stages[i]((_p, _f, indices[i], offset[i : i + 2]))
            p.append(_p)
            f.append(_f)

        for i in range(-1, -len(self.dec_stages) - 1, -1):
            f[i - 1] = self.dec_stages[i](
                (p[i - 1], f[i - 1], offset[i - 1]),
                (p[i], f[i], offset[i]),
            )

        return f[1]
