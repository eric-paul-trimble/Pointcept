import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import segment
from pyg_lib.ops import knn
from peft import LoraConfig, get_peft_model
from collections import OrderedDict

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from pointcept.utils.linnet import prepare_linnet_input
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class BoundarySegmentor(nn.Module):
    """Train a segmentation backbone with an optional boundary auxiliary loss."""

    def __init__(
        self,
        backbone=None,
        criteria=None,
        boundary_feat_channels=64,
        boundary_loss_weight=0.3,
        boundary_pos_weight=3.0,
    ):
        super().__init__()

        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.boundary_head = nn.Linear(boundary_feat_channels, 2)
        self.boundary_loss_weight = boundary_loss_weight
        self.register_buffer(
            "boundary_class_weights",
            torch.tensor([1.0, float(boundary_pos_weight)]),
        )

    def forward(self, input_dict):
        if "condition" in input_dict:
            input_dict["condition"] = input_dict["condition"][0]
        if self.training:
            seg_logits, features = self.backbone(input_dict, return_feat=True)
            loss = self.criteria(seg_logits, input_dict["segment"])
            if "boundary" in input_dict:
                boundary_logits = self.boundary_head(features)
                boundary_loss = F.cross_entropy(
                    boundary_logits,
                    input_dict["boundary"].reshape(-1),
                    weight=self.boundary_class_weights,
                )
                loss = loss + self.boundary_loss_weight * boundary_loss

            return {"loss": loss}

        seg_logits = self.backbone(input_dict)
        if "segment" in input_dict:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return {"loss": loss, "seg_logits": seg_logits}

        return {"seg_logits": seg_logits}


@MODELS.register_module()
class DistillSegmentor(nn.Module):
    """Train a student backbone with ground-truth and frozen-teacher targets."""

    def __init__(
        self,
        backbone,
        teacher,
        teacher_weight,
        distill_in_channels,
        num_classes,
        criteria=None,
        distill_loss_weight=0.5,
        teacher_strict=True,
        teacher_out_channels=None,
    ):
        super().__init__()

        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.seg_head = nn.Linear(distill_in_channels, num_classes)
        self.distill_head = nn.Linear(distill_in_channels, num_classes)
        self.distill_loss_weight = distill_loss_weight
        self.teacher_config = teacher
        self.teacher = build_model(teacher)
        self.teacher_head = (
            nn.Sequential(
                nn.BatchNorm1d(teacher_out_channels, momentum=0.02),
                nn.ReLU(),
                nn.Linear(teacher_out_channels, num_classes),
            )
            if teacher_out_channels is not None
            else None
        )
        self._load_teacher(teacher_weight, strict=teacher_strict)
        self.teacher.requires_grad_(False)
        if self.teacher_head is not None:
            self.teacher_head.requires_grad_(False)
        self.teacher.eval()

    def _load_teacher(self, path, strict=True):
        if not path:
            raise ValueError("teacher_weight is required")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError("teacher checkpoint must contain a state dict")
        normalized = {}
        for key, value in state_dict.items():
            key = key.removeprefix("module.")
            key = key.removeprefix("backbone.")
            normalized[key] = value
        if self.teacher_head is None:
            self.teacher.load_state_dict(normalized, strict=strict)
            return

        backbone_state = {}
        teacher_head_state = {}
        for key, value in normalized.items():
            if key.startswith("head."):
                teacher_head_state[f"teacher_head.{key.removeprefix('head.')}"] = value
            else:
                backbone_state[key] = value
        self.teacher.load_state_dict(backbone_state, strict=False)
        if teacher_head_state:
            self.load_state_dict(teacher_head_state, strict=False)

    @staticmethod
    def _logits(output):
        if isinstance(output, dict):
            return output["seg_logits"]

        return output

    @staticmethod
    def _features(output):
        if isinstance(output, tuple):
            return output[-1]

        return output

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Load legacy checkpoints whose segmentation head lived in Volt."""
        state_dict = dict(state_dict)
        for suffix in ("weight", "bias"):
            legacy_key = f"backbone.seg_head.{suffix}"
            current_key = f"seg_head.{suffix}"
            if current_key not in state_dict and legacy_key in state_dict:
                state_dict[current_key] = state_dict[legacy_key]
            state_dict.pop(legacy_key, None)

        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        if self.teacher_head is not None:
            self.teacher_head.eval()

        return self

    def forward(self, input_dict):
        if "condition" in input_dict:
            input_dict["condition"] = input_dict["condition"][0]

        if self.training:
            student_features = self._features(self.backbone(input_dict))
            student_logits = self.seg_head(student_features)
            distill_logits = self.distill_head(student_features)
            with torch.inference_mode():
                teacher_input = prepare_linnet_input(input_dict, self.teacher_config)
                teacher_output = self.teacher(teacher_input)
                teacher_logits = (
                    self.teacher_head(self._features(teacher_output))
                    if self.teacher_head is not None
                    else self._logits(teacher_output)
                )
            loss_seg = self.criteria(student_logits, input_dict["segment"])
            loss_distill = self.criteria(distill_logits, teacher_logits.argmax(-1))
            weight = self.distill_loss_weight
            return {
                "loss": (1.0 - weight) * loss_seg + weight * loss_distill,
            }

        seg_logits = self.seg_head(self._features(self.backbone(input_dict)))
        if "segment" in input_dict:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return {"loss": loss, "seg_logits": seg_logits}

        return {"seg_logits": seg_logits}


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultLORASegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        backbone_path=None,
        keywords=None,
        replacements=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.keywords = keywords
        self.replacements = replacements
        self.backbone = build_model(backbone)
        backbone_weight = torch.load(
            backbone_path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        self.backbone_load(backbone_weight)

        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                # target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        self.backbone.enc.print_trainable_parameters()

    def backbone_load(self, checkpoint):
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            if self.keywords in key:
                key = key.replace(self.keywords, self.replacements)
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
            if key.startswith("backbone."):
                key = key[9:]
            weight[key] = value
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        print(f"Missing keys: {load_state_info[0]}")
        print(f"Unexpected keys: {load_state_info[1]}")

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.freeze_backbone and not self.use_lora:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = knn(
            x=dino_coord,
            y=point.origin_coord,
            ptr_x=dino_offset,
            ptr_y=point.origin_offset,
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["origin_segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["origin_segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.freeze_backbone = freeze_backbone
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, input_dict):
        point = Point(input_dict)
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = segment(
                point.feat,
                nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
