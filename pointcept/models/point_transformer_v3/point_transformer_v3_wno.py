"""Point Transformer V3 with a shared Wavelet Neural Operator branch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_utils

from pointcept.models.builder import MODELS

from pointcept.models.modules import PointModule
from .point_transformer_v3m1_base import PointTransformerV3

if TYPE_CHECKING:
    from pointcept.models.utils.structure import Point


class WaveletNeuralOperator(nn.Module):
    """Trainable, learnable-basis wavelet operator on a dense 3D feature grid.

    Multi-level analysis (depthwise strided conv) -> coarsest-scale channel mixing
    (1x1x1 conv = R_theta) -> multi-level synthesis (depthwise transposed conv),
    plus a depthwise local bypass B preserving high-frequency detail (paper Eq. 8).
    Shared across all PTv3 stages, so the parameter footprint stays tiny.
    """

    def __init__(
        self, dim: int = 64, levels: int = 2, act_layer: type[nn.Module] = nn.GELU
    ) -> None:
        super().__init__()

        self.levels = levels
        # analysis: depthwise stride-2 conv (G -> G/2), one per level
        self.analysis = nn.ModuleList(
            [
                nn.Conv3d(dim, dim, kernel_size=2, stride=2, groups=dim)
                for _ in range(levels)
            ]
        )
        # synthesis: depthwise stride-2 transposed conv (G/2 -> G), one per level
        self.synthesis = nn.ModuleList(
            [
                nn.ConvTranspose3d(dim, dim, kernel_size=2, stride=2, groups=dim)
                for _ in range(levels)
            ]
        )
        # R_theta: 1x1x1 conv mixing sub-band coefficients across channels
        self.mix = nn.Conv3d(dim, dim, kernel_size=1)
        # B: local depthwise bypass (high-frequency / identity-ish path)
        self.bypass = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = act_layer()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        h = u
        for a in self.analysis:
            h = self.act(a(h))
        h = self.mix(h)
        for s in self.synthesis:
            h = self.act(s(h))

        return h + self.bypass(u)


class WNOFusion(PointModule):
    """Per-stage WNO branch: project -> voxelize -> (shared) WNO -> lift -> fuse.

    Reads and writes ``point.feat`` so subsequent PTv3 stages consume fused
    per-point features directly.
    """

    def __init__(
        self,
        channels: int,
        wno: WaveletNeuralOperator,
        pos_mlp: nn.Module,
        dim: int = 64,
        grid: int = 64,
        eps: float = 1e-6,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.grid = grid
        self.eps = eps
        self.use_checkpoint = use_checkpoint
        self.wno = wno  # shared WaveletNeuralOperator
        self.pos_mlp = pos_mlp  # shared MLP_pos (4 -> dim)
        self.in_proj = nn.Sequential(
            nn.Linear(channels, dim), nn.GELU()
        )  # MLP_in (C -> D)
        self.out_proj = nn.Linear(dim, channels)  # MLP_out (D -> C)
        self.fuse = nn.Sequential(  # phi(W [h || g])
            nn.Linear(2 * channels, channels), nn.LayerNorm(channels), nn.GELU()
        )

    def _voxel_index_and_pos(
        self, coord: torch.Tensor, batch: torch.Tensor, num_scenes: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Per-scene voxel linear index (offset by batch) and normalized pos feats."""
        G = self.grid
        mins = coord.new_full((num_scenes, 3), torch.inf)
        maxs = coord.new_full((num_scenes, 3), -torch.inf)
        mins.index_reduce_(0, batch, coord, "amin", include_self=True)
        maxs.index_reduce_(0, batch, coord, "amax", include_self=True)

        scene_min = mins[batch]
        scene_max = maxs[batch]
        voxel = torch.floor(
            (coord - scene_min) / (scene_max - scene_min + self.eps) * (G - 1)
        ).long()
        voxel = voxel.clamp(0, G - 1)
        lin = (voxel[:, 0] * G * G + voxel[:, 1] * G + voxel[:, 2]) + batch * (G**3)

        counts = torch.bincount(batch, minlength=num_scenes).to(coord.dtype)
        centroid = coord.new_zeros(num_scenes, 3)
        centroid.index_add_(0, batch, coord)
        centroid = centroid / counts.clamp_min(1).unsqueeze(1)
        delta = coord - centroid[batch]
        radius = torch.linalg.norm(delta, dim=1)
        dnorm = delta / (delta.abs().amax(dim=0) + self.eps)
        rnorm = radius / radius.amax().clamp_min(self.eps)
        pos = torch.cat([dnorm, rnorm.unsqueeze(1)], dim=1)

        return lin, pos, num_scenes

    def forward(self, point: Point) -> Point:
        feat = point.feat  # (N, C)
        coord = point.coord  # (N, 3)
        batch = point.batch  # (N,)
        G, D = self.grid, self.dim
        in_dtype = feat.dtype

        # The whole WNO branch runs in fp32 with autocast DISABLED. Under AMP the
        # grid projection mean-pools many points per voxel via index_add_; in dense
        # regions that running sum overflows fp16 (max ~6.5e4) -> inf/NaN. The branch
        # is a tiny fraction of compute (convs on a 64^3 grid + per-point linears),
        # so fp32 here costs almost nothing but makes training numerically stable.
        with torch.autocast(device_type=feat.device.type, enabled=False):
            f32 = feat.float()
            lin, pos, B = self._voxel_index_and_pos(
                coord.float(), batch, point.offset.shape[0]
            )

            def _branch(f32: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
                h = self.in_proj(f32)  # (N, D)
                h = h + self.pos_mlp(pos)  # + relative position encoding

                # grid projection: mean-pool features into occupied voxels (Eq. 5)
                flat = B * (G**3)
                acc = torch.zeros(flat, D, dtype=torch.float32, device=h.device)
                cnt = torch.zeros(flat, 1, dtype=torch.float32, device=h.device)
                acc.index_add_(0, lin, h)
                cnt.index_add_(0, lin, torch.ones_like(h[:, :1]))
                u = (
                    (acc / cnt.clamp(min=1.0))
                    .view(B, G, G, G, D)
                    .permute(0, 4, 1, 2, 3)
                    .contiguous()
                )

                # shared wavelet neural operator (Eq. 8)
                v = self.wno(u)  # (B, D, G, G, G)

                # grid-to-point lifting (Eq. 9)
                v_flat = v.permute(0, 2, 3, 4, 1).reshape(flat, D)
                g = self.out_proj(v_flat[lin])  # (N, C)

                # fusion (Eq. 10): replaces h for subsequent blocks
                return self.fuse(torch.cat([f32, g], dim=1))

            # The dense GxGxG grid + 3D (transposed) convs make this branch the second
            # memory hog after the conv; checkpoint it so the grid activations are
            # recomputed in backward instead of stored (kept across every stage).
            if self.use_checkpoint and self.training and f32.requires_grad:
                fused = checkpoint_utils.checkpoint(
                    _branch, f32, pos, use_reentrant=False
                )
            else:
                fused = _branch(f32, pos)

        point.feat = fused.to(in_dtype)

        return point


@MODELS.register_module("PT-WNO")
class PointTransformerV3WNO(PointTransformerV3):
    """PTv3 backbone with a shared Wavelet Neural Operator branch fused per stage."""

    def __init__(
        self,
        *args: Any,
        wno_dim: int = 64,
        wno_grid: int = 64,
        wno_levels: int = 2,
        wno_enc_stages: Sequence[int] | None = None,
        wno_dec_stages: Sequence[int] | None = None,
        use_checkpoint: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        enc_channels = kwargs.get("enc_channels", (32, 64, 128, 256, 512))
        dec_channels = kwargs.get("dec_channels", (64, 64, 128, 256))
        enc_depths = kwargs.get("enc_depths", (2, 2, 2, 6, 2))
        num_stages = len(enc_depths)

        # which stages get a WNO branch (default: all of them)
        if wno_enc_stages is None:
            wno_enc_stages = list(range(num_stages))
        if wno_dec_stages is None:
            wno_dec_stages = list(range(num_stages - 1))

        # shared operator + shared relative-position MLP (paper: K_theta shared)
        self.wno = WaveletNeuralOperator(dim=wno_dim, levels=wno_levels)
        self.pos_mlp = nn.Sequential(
            nn.Linear(4, wno_dim),
            nn.LayerNorm(wno_dim),
            nn.GELU(),
            nn.Linear(wno_dim, wno_dim),
        )

        for s in wno_enc_stages:
            name = f"enc{s}"
            if name in self.enc._modules:
                self.enc._modules[name].add(
                    WNOFusion(
                        enc_channels[s],
                        self.wno,
                        self.pos_mlp,
                        wno_dim,
                        wno_grid,
                        use_checkpoint=use_checkpoint,
                    ),
                    name="wno",
                )
        if not self.cls_mode and hasattr(self, "dec"):
            for s in wno_dec_stages:
                name = f"dec{s}"
                if name in self.dec._modules:
                    self.dec._modules[name].add(
                        WNOFusion(
                            dec_channels[s],
                            self.wno,
                            self.pos_mlp,
                            wno_dim,
                            wno_grid,
                            use_checkpoint=use_checkpoint,
                        ),
                        name="wno",
                    )
