"""LinNet input preparation shared by training, testing, and models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyg_lib.ops as pyg_ops
import torch

from pointcept.utils.offset import batch2offset, offset2batch


def linearization_sampling(
    coord: torch.Tensor, offset: torch.Tensor, grid_size: float = 0.01
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one point per voxel and return its indices and batch offsets."""
    if coord.numel() == 0:
        raise ValueError("Cannot sample an empty point cloud")

    shifted_coord = coord - coord.amin(dim=0)
    size = coord.new_tensor([grid_size, grid_size, grid_size])
    batch = offset2batch(offset)
    global_extent = shifted_coord.amax(dim=0)
    stride = (torch.ceil(global_extent / grid_size) + 2) * grid_size
    center = shifted_coord + batch.unsqueeze(1).to(coord.dtype) * stride

    relative = torch.remainder(shifted_coord, size[0]).square().sum(dim=-1)
    cluster = pyg_ops.grid_cluster(center, size)
    sorted_cluster, order = torch.sort(cluster)
    _, _, counts = torch.unique(
        sorted_cluster, sorted=True, return_inverse=True, return_counts=True
    )
    indptr = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)])
    _, representative = pyg_ops.segment_min_csr(relative[order], indptr)
    index = order[representative]
    sampled_batch = batch[index]

    return index.contiguous(), batch2offset(sampled_batch).contiguous()


def _limit_stage_indices(
    index: torch.Tensor, source_offset: torch.Tensor, max_points: int, min_points: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Limit each batch element while keeping enough points for LinNet KNN."""
    source_batch = offset2batch(source_offset)
    sampled_batch = source_batch[index]
    selected: list[torch.Tensor] = []
    for batch_id in range(int(sampled_batch.max().item()) + 1):
        batch_index = torch.where(sampled_batch == batch_id)[0]
        if batch_index.numel() == 0:
            continue

        count = batch_index.numel()
        keep_count = min(count, max_points)
        permutation = torch.randperm(count, device=index.device)[:keep_count]
        keep = batch_index[permutation]
        desired = min(max_points, max(min_points, count))
        if keep.numel() < desired:
            repeats = torch.randint(
                keep.numel(), (desired - keep.numel(),), device=index.device
            )
            keep = torch.cat([keep, keep[repeats]])
        selected.append(index[keep])

    if not selected:
        raise ValueError("Sampling produced no points")

    index = torch.cat(selected).contiguous()

    return index, batch2offset(source_batch[index]).contiguous()


def pre_sample(
    input_dict: dict,
    grid_sizes: Sequence[float],
    point_counts: Sequence[int] | None = None,
    min_pts: int = 24,
) -> dict:
    """Prepare the multiscale indices and offsets consumed by LinNet."""
    if "offset1" in input_dict or "indices" in input_dict:
        if not {"offset1", "indices"}.issubset(input_dict):
            raise ValueError("LinNet preparation requires both offset1 and indices")

        return input_dict

    coord = input_dict["coord"]
    source_offset = input_dict["offset"]
    offsets = [source_offset]
    indices: list[torch.Tensor] = []
    current_coord = coord
    current_offset = source_offset

    limits = None if point_counts is None else list(point_counts)
    if limits is not None and len(limits) != len(grid_sizes):
        raise ValueError("point_counts and grid_sizes must have equal lengths")
    if input_dict.get("Mix3d", False) and limits is not None:
        limits = [count * 2 for count in limits]

    for stage, grid_size in enumerate(grid_sizes):
        index, next_offset = linearization_sampling(
            current_coord, current_offset, float(grid_size)
        )
        if limits is not None:
            index, next_offset = _limit_stage_indices(
                index, current_offset, int(limits[stage]), min_pts
            )
        current_coord = current_coord[index]
        current_offset = next_offset
        indices.append(index.long())
        offsets.append(current_offset)

    input_dict["offset1"] = offsets
    input_dict["indices"] = indices

    return input_dict


def _find_linnet_config(model_cfg: Mapping | None) -> Mapping | None:
    """Find a LinNet configuration in a model or distillation configuration."""
    if model_cfg is None:
        return None

    if model_cfg.get("type") == "LinNet":
        return model_cfg

    for key in ("backbone", "teacher"):
        nested = model_cfg.get(key)
        found = _find_linnet_config(nested)
        if found is not None:
            return found

    return None


def prepare_linnet_input(input_dict: dict, model_cfg: Mapping | None) -> dict:
    """Prepare LinNet inputs once for training, evaluation, testing, or inference."""
    linnet_cfg = _find_linnet_config(model_cfg)
    if linnet_cfg is None:
        return input_dict

    grid_sizes = linnet_cfg.get("grid_sizes")
    if not grid_sizes:
        raise ValueError("LinNet requires non-empty grid_sizes")

    return pre_sample(
        input_dict,
        grid_sizes,
        point_counts=linnet_cfg.get("point_counts"),
        min_pts=int(linnet_cfg.get("nsample", 24)),
    )
