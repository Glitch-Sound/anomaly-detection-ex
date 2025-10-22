import configparser
import csv
import importlib
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

_albucore_utils = None
try:
    _albucore_utils = importlib.import_module("albucore.utils")
except ModuleNotFoundError:
    pass
else:
    if not hasattr(_albucore_utils, "preserve_channel_dim"):
        from functools import wraps

        def _preserve_channel_dim(func):
            @wraps(func)
            def wrapper(image, *args, **kwargs):
                original_shape = getattr(image, "shape", None)
                result = func(image, *args, **kwargs)
                if original_shape is None:
                    return result
                if len(original_shape) == 3 and result.ndim == 2:
                    return result[:, :, np.newaxis]
                return result

            return wrapper

        setattr(_albucore_utils, "preserve_channel_dim", _preserve_channel_dim)
    if not hasattr(_albucore_utils, "MONO_CHANNEL_DIMENSIONS"):
        setattr(_albucore_utils, "MONO_CHANNEL_DIMENSIONS", 2)
    if not hasattr(_albucore_utils, "NUM_MULTI_CHANNEL_DIMENSIONS"):
        setattr(_albucore_utils, "NUM_MULTI_CHANNEL_DIMENSIONS", 3)

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# =========================
# ハイパーパラメータ定義
# =========================
MODEL_VARIANT_DEFAULT = "deit_base_distilled_patch16_384"
MODEL_VARIANTS = {
    "deit_base_distilled_patch16_384": {
        "image_size": 384,
        "layer_indices": [11],
    },
    "deit_base_distilled_patch16_224": {
        "image_size": 224,
        "layer_indices": [11],
    },
}
PATCH_SIZE = 16
TRAIN_BATCH_SIZE = 4
TRAIN_EPOCHS = 10
FLOW_STEPS = 8
FLOW_HIDDEN_RATIO = 1.0
LEARNING_RATE = 1e-3
THRESHOLD_QUANTILE = 0.995
NO_HEATMAP_MARGIN = 5.0
NUM_WORKERS = 0
CONFIG_SECTION = "AI"
CONFIG_FIELDS = {
    "PATCH_SIZE": ("PATCH_SIZE", int),
    "TRAIN_BATCH_SIZE": ("TRAIN_BATCH_SIZE", int),
    "TRAIN_EPOCHS": ("TRAIN_EPOCHS", int),
    "FLOW_STEPS": ("FLOW_STEPS", int),
    "FLOW_HIDDEN_RATIO": ("FLOW_HIDDEN_RATIO", float),
    "LEARNING_RATE": ("LEARNING_RATE", float),
    "THRESHOLD_QUANTILE": ("THRESHOLD_QUANTILE", float),
    "NO_HEATMAP_MARGIN": ("NO_HEATMAP_MARGIN", float),
    "NUM_WORKERS": ("NUM_WORKERS", int),
}
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_cache_dir = str(MODEL_DIR.resolve())
os.environ.setdefault("TIMM_MODEL_CACHE", model_cache_dir)
os.environ.setdefault("TORCH_HOME", model_cache_dir)
os.environ.setdefault("HF_HOME", model_cache_dir)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", model_cache_dir)
try:
    from timm.models import hub as timm_hub
except ImportError:
    timm_hub = None
else:
    setter = getattr(timm_hub, "set_model_cache_dir", None)
    if callable(setter):
        setter(model_cache_dir)
    else:
        setter_alt = getattr(timm_hub, "set_default_cache_dir", None)
        if callable(setter_alt):
            setter_alt(model_cache_dir)
LOG_DIR = Path("log")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LOG_PATH = LOG_DIR / "training.csv"
DETECTION_LOG_PATH = LOG_DIR / "detection.csv"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]
    )


def list_image_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return files


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[Path],
        transform: A.Compose,
        *,
        return_meta: bool = False,
    ) -> None:
        if not paths:
            raise ValueError("画像が見つかりません。パスを確認してください。")
        self.paths = list(paths)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"画像を読み込めません: {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed["image"].float()
        if self.return_meta:
            return image_tensor, image_bgr, str(path)
        return image_tensor


class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_variant: str, layer_indices: Sequence[int]):
        super().__init__()
        self.model = timm.create_model(model_variant, pretrained=True)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        self.embed_dim = getattr(self.model, "embed_dim")
        base_tokens = getattr(self.model, "num_tokens", 1)
        has_dist = getattr(self.model, "dist_token", None) is not None
        self.token_offset = base_tokens + (1 if has_dist else 0)
        total_blocks = len(self.model.blocks)
        if total_blocks <= 0:
            raise ValueError("ViT モデルにブロックが存在しません。")
        if not layer_indices:
            layer_indices = [total_blocks - 1]
        normalized: List[int] = []
        for idx in sorted(set(layer_indices)):
            if idx < 0:
                idx = total_blocks + idx
            if idx < 0 or idx >= total_blocks:
                raise ValueError(
                    f"layer index {idx} は許容範囲外です (0-{total_blocks - 1})"
                )
            normalized.append(idx)
        self.layer_indices = tuple(normalized)
        self.layer_index_set = set(self.layer_indices)
        self.output_dim = self.embed_dim * len(self.layer_indices)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        model = self.model
        x = model.patch_embed(images)
        batch = x.shape[0]
        cls_token = model.cls_token.expand(batch, -1, -1)
        if getattr(model, "dist_token", None) is not None:
            dist_token = model.dist_token.expand(batch, -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)
        x = model.pos_drop(x + model.pos_embed)
        collected: List[torch.Tensor] = []
        for idx, block in enumerate(model.blocks):
            x = block(x)
            if idx in self.layer_index_set:
                normed = model.norm(x)
                collected.append(normed[:, self.token_offset :, :])
        if not collected:
            normed = model.norm(x)
            collected.append(normed[:, self.token_offset :, :])
        return torch.cat(collected, dim=2)


def _append_log_row(path: Path, header: Sequence[str], row: Sequence) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as fp:
        writer = csv.writer(fp)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def log_training_run(artifacts: "TrainingArtifacts", dataset_size: int) -> None:
    header = [
        "timestamp",
        "model_variant",
        "layer_indices",
        "dataset_size",
        "train_epochs",
        "batch_size",
        "learning_rate",
        "flow_steps",
        "flow_hidden_ratio",
        "threshold",
        "quantile",
        "num_patches",
        "embed_dim",
        "avg_patch_mean",
        "avg_patch_std",
        "no_heatmap_margin",
    ]
    avg_patch_mean = float(np.mean(artifacts.patch_means)) if artifacts.patch_means else 0.0
    avg_patch_std = float(np.mean(artifacts.patch_stds)) if artifacts.patch_stds else 0.0
    layer_indices_str = "|".join(map(str, artifacts.layer_indices))
    row = [
        datetime.utcnow().isoformat(timespec="seconds"),
        artifacts.model_variant,
        layer_indices_str,
        dataset_size,
        TRAIN_EPOCHS,
        TRAIN_BATCH_SIZE,
        LEARNING_RATE,
        artifacts.flow_steps,
        artifacts.flow_hidden_ratio,
        artifacts.threshold,
        artifacts.quantile,
        artifacts.num_patches,
        artifacts.embed_dim,
        avg_patch_mean,
        avg_patch_std,
        NO_HEATMAP_MARGIN,
    ]
    _append_log_row(TRAIN_LOG_PATH, header, row)


def log_detection_result(
    artifacts: "TrainingArtifacts",
    image_path: Path,
    label: str,
    predicted: str,
    score_max_norm: float,
    score_mean_norm: float,
    score_max_raw: float,
    score_mean_raw: float,
    skip_heatmap: bool,
) -> None:
    header = [
        "timestamp",
        "image_path",
        "label",
        "predicted",
        "score_max_norm",
        "score_mean_norm",
        "score_max_raw",
        "score_mean_raw",
        "threshold",
        "quantile",
        "model_variant",
        "layer_indices",
        "num_patches",
        "embed_dim",
        "flow_steps",
        "flow_hidden_ratio",
        "no_heatmap_margin",
        "skip_heatmap",
    ]
    row = [
        datetime.utcnow().isoformat(timespec="seconds"),
        str(image_path),
        label,
        predicted,
        score_max_norm,
        score_mean_norm,
        score_max_raw,
        score_mean_raw,
        artifacts.threshold,
        artifacts.quantile,
        artifacts.model_variant,
        "|".join(map(str, artifacts.layer_indices)),
        artifacts.num_patches,
        artifacts.embed_dim,
        artifacts.flow_steps,
        artifacts.flow_hidden_ratio,
        NO_HEATMAP_MARGIN,
        skip_heatmap,
    ]
    _append_log_row(DETECTION_LOG_PATH, header, row)


class CouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float) -> None:
        super().__init__()
        hidden_dim = max(32, int(dim * hidden_ratio))
        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x.chunk(2, dim=1)
        s, t = self.net(x1).chunk(2, dim=1)
        s = torch.tanh(s)
        if not reverse:
            y2 = x2 * torch.exp(s) + t
            logdet = logdet + s.sum(dim=1)
        else:
            y2 = (x2 - t) * torch.exp(-s)
            logdet = logdet - s.sum(dim=1)
        y = torch.cat([x1, y2], dim=1)
        return y, logdet


class PermuteLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        perm = torch.randperm(dim)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(dim)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv)

    def forward(
        self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            return x[:, self.inv_perm], logdet
        return x[:, self.perm], logdet


class FastFlowNF(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float, steps: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("FastFlow の入力次元は偶数である必要があります。")
        layers: List[nn.Module] = []
        for _ in range(steps):
            layers.append(PermuteLayer(dim))
            layers.append(CouplingLayer(dim, hidden_ratio))
        self.layers = nn.ModuleList(layers)
        self.dim = dim

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        if not reverse:
            for layer in self.layers:
                x, logdet = layer(x, logdet, reverse=False)
            return x, logdet
        for layer in reversed(self.layers):
            x, logdet = layer(x, logdet, reverse=True)
        return x, logdet

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, logdet = self.forward(x, reverse=False)
        log_base = -0.5 * torch.sum(z * z, dim=1) - 0.5 * self.dim * math.log(
            2 * math.pi
        )
        return log_base + logdet

    def anomaly_scores(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(-1, self.dim)
        log_prob = self.log_prob(flat)
        scores = (-log_prob).reshape(features.shape[0], features.shape[1])
        return scores


@dataclass
class TrainingArtifacts:
    flow_state_dict: dict
    model_variant: str
    image_size: int
    patch_size: int
    num_patches: int
    threshold: float
    quantile: float
    embed_dim: int
    flow_steps: int
    flow_hidden_ratio: float
    layer_indices: List[int] = field(default_factory=list)
    patch_means: List[float] = field(default_factory=list)
    patch_stds: List[float] = field(default_factory=list)


def set_config(path_config: str) -> None:
    config_path = Path(path_config)
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    parser = configparser.ConfigParser()
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            parser.read_file(fp)
    except Exception as exc:  # noqa: PERF203
        raise RuntimeError(f"設定ファイルの読み込みに失敗しました: {config_path}") from exc

    if CONFIG_SECTION not in parser:
        raise KeyError(f"設定ファイルにセクション[{CONFIG_SECTION}]が存在しません。")

    section = parser[CONFIG_SECTION]

    def _cast(value: str, caster):
        text = value.strip()
        if caster is int:
            return int(text)
        if caster is float:
            return float(text)
        return caster(text)

    global_vars = globals()
    for key, (var_name, caster) in CONFIG_FIELDS.items():
        if key not in section or section[key] == "":
            continue
        raw_value = section[key]
        try:
            converted = _cast(raw_value, caster)
        except ValueError as exc:
            raise ValueError(
                f"設定値 {key} の形式が不正です: {raw_value}"
            ) from exc
        global_vars[var_name] = converted


def train(path_config: str, path_train_good: str, path_param: str) -> None:
    set_config(path_config)
    device = get_device()
    model_variant = MODEL_VARIANT_DEFAULT
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"サポート外のモデルです: {model_variant}")
    variant_conf = MODEL_VARIANTS[model_variant]
    image_size = variant_conf["image_size"]
    layer_indices = variant_conf.get("layer_indices", [])
    transform = build_transform(image_size)
    good_paths = list_image_files(Path(path_train_good))
    dataset = ImageFolderDataset(good_paths, transform, return_meta=False)
    dataset_size = len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    extractor = ViTFeatureExtractor(model_variant, layer_indices).to(device)
    flow_model = FastFlowNF(
        extractor.output_dim, FLOW_HIDDEN_RATIO, FLOW_STEPS
    ).to(device)
    optimizer = Adam(flow_model.parameters(), lr=LEARNING_RATE)
    flow_model.train()
    num_patches: int | None = None
    for epoch in range(TRAIN_EPOCHS):
        epoch_loss = 0.0
        batch_count = 0
        for images in loader:
            images = images.to(device)
            with torch.no_grad():
                tokens = extractor(images)
            if num_patches is None:
                num_patches = tokens.shape[1]
            tokens = tokens.detach()
            features = tokens.reshape(-1, extractor.output_dim)
            log_prob = flow_model.log_prob(features)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"epoch {epoch+1}/{TRAIN_EPOCHS} loss={avg_loss:.6f}")
    flow_model.eval()
    patch_scores: List[torch.Tensor] = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            tokens = extractor(images)
            if num_patches is None:
                num_patches = tokens.shape[1]
            anomaly = flow_model.anomaly_scores(tokens)
            patch_scores.append(anomaly.cpu())
    if not patch_scores:
        raise ValueError("学習データが存在しないため学習を継続できません。")
    patch_scores_tensor = torch.cat(patch_scores, dim=0)
    patch_means = patch_scores_tensor.mean(dim=0)
    patch_stds = patch_scores_tensor.std(dim=0)
    patch_stds = torch.where(
        patch_stds < 1e-6, torch.full_like(patch_stds, 1e-6), patch_stds
    )
    standardized_scores = (patch_scores_tensor - patch_means) / patch_stds
    per_image_scores = standardized_scores.max(dim=1).values
    threshold = float(torch.quantile(per_image_scores, THRESHOLD_QUANTILE))
    if num_patches is None:
        raise RuntimeError("パッチ数の取得に失敗しました。")
    artifacts = TrainingArtifacts(
        flow_state_dict=flow_model.state_dict(),
        model_variant=model_variant,
        image_size=image_size,
        patch_size=PATCH_SIZE,
        num_patches=num_patches,
        threshold=threshold,
        quantile=THRESHOLD_QUANTILE,
        embed_dim=extractor.output_dim,
        flow_steps=FLOW_STEPS,
        flow_hidden_ratio=FLOW_HIDDEN_RATIO,
        layer_indices=list(layer_indices),
        patch_means=patch_means.tolist(),
        patch_stds=patch_stds.tolist(),
    )
    param_path = Path(path_param)
    param_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts.__dict__, param_path)
    print(f"model saved to {param_path}")
    log_training_run(artifacts, dataset_size)


def test(path_config: str, path_test: str, path_param: str, path_result: str) -> None:
    set_config(path_config)
    param_path = Path(path_param)
    if not param_path.exists():
        raise FileNotFoundError("学習済みパラメータが存在しません。先に train() を実行してください。")
    state = torch.load(param_path, map_location="cpu")
    if "num_patches" not in state and "image_size" in state and "patch_size" in state:
        approx = (state["image_size"] // state["patch_size"]) ** 2
        state["num_patches"] = approx
    num_patches_state = state.get("num_patches")
    if num_patches_state is None:
        raise RuntimeError("学習済みパラメータにパッチ数が含まれていません。再学習してください。")
    if "layer_indices" not in state or not state["layer_indices"]:
        default_layers = MODEL_VARIANTS.get(
            state.get("model_variant", MODEL_VARIANT_DEFAULT),
            {},
        ).get("layer_indices", [])
        if not default_layers:
            default_layers = [0]
        state["layer_indices"] = default_layers
    if (
        "patch_means" not in state
        or not state["patch_means"]
        or len(state["patch_means"]) != num_patches_state
    ):
        state["patch_means"] = [0.0] * num_patches_state
    if (
        "patch_stds" not in state
        or not state["patch_stds"]
        or len(state["patch_stds"]) != num_patches_state
    ):
        state["patch_stds"] = [1.0] * num_patches_state
    artifacts = TrainingArtifacts(**state)
    image_size = artifacts.image_size
    transform = build_transform(image_size)
    test_root = Path(path_test)
    good_test = list_image_files(test_root / "good")
    error_test = list_image_files(test_root / "error")
    all_paths = [(p, "good") for p in good_test] + [(p, "error") for p in error_test]
    if not all_paths:
        raise ValueError("試験用画像が見つかりません。")
    dataset = ImageFolderDataset([p for p, _ in all_paths], transform, return_meta=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    device = get_device()
    extractor = ViTFeatureExtractor(artifacts.model_variant, artifacts.layer_indices).to(device)
    flow_model = FastFlowNF(
        artifacts.embed_dim, artifacts.flow_hidden_ratio, artifacts.flow_steps
    ).to(device)
    flow_model.load_state_dict(artifacts.flow_state_dict)
    flow_model.eval()
    patch_means_tensor = (
        torch.tensor(artifacts.patch_means, device=device, dtype=torch.float32)
        .clamp(min=-1e9, max=1e9)
        .unsqueeze(0)
    )
    patch_stds_tensor = (
        torch.tensor(artifacts.patch_stds, device=device, dtype=torch.float32)
        .clamp_min(1e-6)
        .unsqueeze(0)
    )
    result_root = Path(path_result)
    (result_root / "good").mkdir(parents=True, exist_ok=True)
    (result_root / "error").mkdir(parents=True, exist_ok=True)
    patch_side = int(round(math.sqrt(artifacts.num_patches)))
    if patch_side * patch_side != artifacts.num_patches:
        raise ValueError("パッチ数が正方格子にマッピングできません。")
    labels = [label for _, label in all_paths]
    with torch.no_grad():
        for idx, (image_tensor, original, paths) in enumerate(loader):
            label = labels[idx]
            image_tensor = image_tensor.to(device)
            tokens = extractor(image_tensor)
            anomaly_map = flow_model.anomaly_scores(tokens)
            normalized_map = torch.clamp(
                (anomaly_map - patch_means_tensor) / patch_stds_tensor, min=0.0
            )
            raw_patch_map = anomaly_map.reshape(1, patch_side, patch_side)
            norm_patch_map = normalized_map.reshape(1, patch_side, patch_side)
            score_map = (
                F.interpolate(
                    norm_patch_map.unsqueeze(1),
                    size=image_tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            image_score_norm = float(normalized_map.max().item())
            score_mean_norm = float(normalized_map.mean().item())
            image_score_raw = float(anomaly_map.max().item())
            score_mean_raw = float(anomaly_map.mean().item())
            predicted = "error" if image_score_norm > artifacts.threshold else "good"
            skip_heatmap = abs(image_score_norm - artifacts.threshold) <= NO_HEATMAP_MARGIN
            save_dir = result_root / label
            save_dir.mkdir(parents=True, exist_ok=True)
            path_obj = Path(paths[0])
            suffix = "_heatmap.png"
            if skip_heatmap:
                suffix = "_original.png"
            save_path = save_dir / f"{path_obj.stem}{suffix}"
            original_image = np.asarray(original)[0]
            if skip_heatmap:
                cv2.imwrite(str(save_path), original_image)
            else:
                overlay = create_heatmap_overlay(original_image, score_map)
                cv2.imwrite(str(save_path), overlay)
            print(
                f"{path_obj.name}: score_norm={image_score_norm:.4f} raw={image_score_raw:.4f} threshold={artifacts.threshold:.4f} predict={predicted} skip_heatmap={skip_heatmap}"
            )
            log_detection_result(
                artifacts,
                path_obj,
                label,
                predicted,
                image_score_norm,
                score_mean_norm,
                image_score_raw,
                score_mean_raw,
                skip_heatmap,
            )


def create_heatmap_overlay(image_bgr: np.ndarray, score_map: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    norm_map = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(norm_map.astype(np.uint8), cv2.COLORMAP_JET)
    resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(resized, alpha, image_bgr, 1 - alpha, 0)
    return overlay


if __name__ == "__main__":
    try:
        path_config = "config/setting.ini"
        path_train = "data/train/good"
        path_test = "data/test"
        path_param = "param/param.pth"
        path_result = "data/result"
        print(sys.version)

        print("start train.")
        train(path_config, path_train, path_param)

        print("start test.")
        test(path_config, path_test, path_param, path_result)

    except Exception:
        traceback.print_exc()
        sys.exit(1)

    finally:
        sys.exit(0)
