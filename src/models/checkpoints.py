from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src import config

REGISTRY_FILENAME = "registry.json"
BEST_FILENAME = "best.pt"


@dataclass
class CheckpointInfo:
    run_id: str
    path: Path
    metrics: dict[str, float]
    epoch: int

    @property
    def val_acc(self) -> float:
        return float(self.metrics.get("val_acc", 0.0))

    @property
    def val_loss(self) -> float:
        return float(self.metrics.get("val_loss", float("inf")))


def checkpoints_root() -> Path:
    return config.PROJECT_ROOT / "models" / "checkpoints"


def model_dir(model_name: str) -> Path:
    return checkpoints_root() / model_name


def _registry_path(model_name: str) -> Path:
    return model_dir(model_name) / REGISTRY_FILENAME


def _best_path(model_name: str) -> Path:
    return model_dir(model_name) / BEST_FILENAME


def _load_registry(model_name: str) -> dict[str, Any] | None:
    path = _registry_path(model_name)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_registry(model_name: str, entry: dict[str, Any]) -> None:
    model_dir(model_name).mkdir(parents=True, exist_ok=True)
    with open(_registry_path(model_name), "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)


def get_best_checkpoint(model_name: str) -> CheckpointInfo | None:
    registry = _load_registry(model_name)
    best_path = _best_path(model_name)
    if registry is None or not best_path.exists():
        return None
    return CheckpointInfo(
        run_id=registry["run_id"],
        path=best_path,
        metrics=registry.get("metrics", {}),
        epoch=int(registry.get("epoch", 0)),
    )


def list_versions(model_name: str) -> list[CheckpointInfo]:
    """Returns at most one entry — only the best weights are kept."""
    info = get_best_checkpoint(model_name)
    return [info] if info else []


def save_best(
    model_name: str,
    model: nn.Module,
    *,
    optimizer: Optimizer | None = None,
    epoch: int,
    metrics: dict[str, float],
    history: dict[str, list[float]] | None = None,
    extra: dict[str, Any] | None = None,
) -> CheckpointInfo:
    """Overwrite the single best checkpoint for this model."""
    model_dir(model_name).mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = _best_path(model_name)

    payload: dict[str, Any] = {
        "model_name": model_name,
        "run_id": run_id,
        "epoch": epoch,
        "metrics": metrics,
        "history": history or {},
        "model_state_dict": model.state_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, path)

    entry = {
        "run_id": run_id,
        "path": path.relative_to(config.PROJECT_ROOT).as_posix(),
        "epoch": epoch,
        "metrics": metrics,
        "created_at": payload["created_at"],
    }
    _save_registry(model_name, entry)

    print(
        f"Saved best {model_name} (epoch {epoch}) "
        f"val_loss={metrics.get('val_loss', 0):.4f} "
        f"val_acc={metrics.get('val_acc', 0):.4f} -> {path}"
    )
    return CheckpointInfo(
        run_id=run_id,
        path=path,
        metrics=metrics,
        epoch=epoch,
    )


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path

    location = map_location if map_location is not None else config.get_device()
    payload = torch.load(path, map_location=location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def load_best(
    model_name: str,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> dict[str, Any] | None:
    info = get_best_checkpoint(model_name)
    if info is None:
        return None
    payload = load_checkpoint(info.path, model, optimizer)
    print(
        f"Loaded best {model_name} (epoch={info.epoch}, "
        f"val_loss={info.val_loss:.4f}, val_acc={info.val_acc:.4f})"
    )
    return payload


def load_latest(
    model_name: str,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> dict[str, Any] | None:
    """Alias for load_best — only the best weights are stored."""
    return load_best(model_name, model, optimizer)
