from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src import config

REGISTRY_FILENAME = "registry.json"
CHECKPOINT_FILENAME = "checkpoint.pt"
BEST_FILENAME = "best.pt"
LATEST_FILENAME = "latest.pt"


@dataclass
class CheckpointInfo:
    version: int
    run_id: str
    path: Path
    metrics: dict[str, float]
    epoch: int

    @property
    def val_acc(self) -> float:
        return float(self.metrics.get("val_acc", 0.0))


def checkpoints_root() -> Path:
    return config.PROJECT_ROOT / "models" / "checkpoints"


def model_dir(model_name: str) -> Path:
    return checkpoints_root() / model_name


def _registry_path(model_name: str) -> Path:
    return model_dir(model_name) / REGISTRY_FILENAME


def _load_registry(model_name: str) -> dict[str, Any]:
    path = _registry_path(model_name)
    if not path.exists():
        return {"versions": [], "best_version": None}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_registry(model_name: str, registry: dict[str, Any]) -> None:
    path = _registry_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _next_version(model_name: str) -> int:
    registry = _load_registry(model_name)
    if not registry["versions"]:
        return 1
    return max(v["version"] for v in registry["versions"]) + 1


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def list_versions(model_name: str) -> list[CheckpointInfo]:
    registry = _load_registry(model_name)
    return [
        CheckpointInfo(
            version=entry["version"],
            run_id=entry["run_id"],
            path=config.PROJECT_ROOT / entry["path"],
            metrics=entry.get("metrics", {}),
            epoch=int(entry.get("epoch", 0)),
        )
        for entry in sorted(registry["versions"], key=lambda e: e["version"])
    ]


def get_best_checkpoint(model_name: str) -> CheckpointInfo | None:
    registry = _load_registry(model_name)
    best_version = registry.get("best_version")
    if best_version is None:
        return None
    for entry in registry["versions"]:
        if entry["version"] == best_version:
            return CheckpointInfo(
                version=entry["version"],
                run_id=entry["run_id"],
                path=config.PROJECT_ROOT / entry["path"],
                metrics=entry.get("metrics", {}),
                epoch=int(entry.get("epoch", 0)),
            )
    return None


def _build_payload(
    model_name: str,
    model: nn.Module,
    *,
    version: int,
    run_id: str,
    optimizer: Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
    history: dict[str, list[float]] | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_name": model_name,
        "version": version,
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
    return payload


def save_latest_snapshot(
    model_name: str,
    model: nn.Module,
    *,
    optimizer: Optimizer | None = None,
    epoch: int,
    metrics: dict[str, float],
    history: dict[str, list[float]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Overwrite latest.pt without bumping version (resume / last epoch)."""
    model_dir(model_name).mkdir(parents=True, exist_ok=True)
    best = get_best_checkpoint(model_name)
    version = best.version if best else 0
    run_id = best.run_id if best else "snapshot"
    payload = _build_payload(
        model_name,
        model,
        version=version,
        run_id=run_id,
        optimizer=optimizer,
        epoch=epoch,
        metrics=metrics,
        history=history,
        extra=extra,
    )
    path = model_dir(model_name) / LATEST_FILENAME
    torch.save(payload, path)
    return path


def save_checkpoint(
    model_name: str,
    model: nn.Module,
    *,
    optimizer: Optimizer | None = None,
    epoch: int,
    metrics: dict[str, float],
    history: dict[str, list[float]] | None = None,
    extra: dict[str, Any] | None = None,
    is_best: bool = False,
) -> CheckpointInfo:
    """Save a versioned checkpoint and update registry + best.pt + latest.pt."""
    version = _next_version(model_name)
    run_id = _new_run_id()
    run_dir = model_dir(model_name) / f"v{version:03d}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / CHECKPOINT_FILENAME
    payload = _build_payload(
        model_name,
        model,
        version=version,
        run_id=run_id,
        optimizer=optimizer,
        epoch=epoch,
        metrics=metrics,
        history=history,
        extra=extra,
    )
    torch.save(payload, checkpoint_path)

    rel_path = checkpoint_path.relative_to(config.PROJECT_ROOT).as_posix()
    entry = {
        "version": version,
        "run_id": run_id,
        "path": rel_path,
        "epoch": epoch,
        "metrics": metrics,
        "created_at": payload["created_at"],
    }

    registry = _load_registry(model_name)
    registry["versions"].append(entry)

    if is_best or registry.get("best_version") is None:
        registry["best_version"] = version
        shutil.copy2(checkpoint_path, model_dir(model_name) / BEST_FILENAME)

    shutil.copy2(checkpoint_path, model_dir(model_name) / LATEST_FILENAME)
    _save_registry(model_name, registry)

    print(
        f"Saved checkpoint v{version:03d} ({model_name}) "
        f"val_acc={metrics.get('val_acc', 0):.4f} -> {checkpoint_path}"
    )
    if is_best:
        print(f"  New best model (v{version:03d})")

    return CheckpointInfo(
        version=version,
        run_id=run_id,
        path=checkpoint_path,
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
    """Load weights into model (and optimizer if provided). Returns full payload."""
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
        f"Loaded best {model_name} v{info.version:03d} "
        f"(val_acc={info.val_acc:.4f}, epoch={info.epoch})"
    )
    return payload


def load_latest(
    model_name: str,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> dict[str, Any] | None:
    latest_path = model_dir(model_name) / LATEST_FILENAME
    if not latest_path.exists():
        return None
    payload = load_checkpoint(latest_path, model, optimizer)
    print(
        f"Loaded latest {model_name} v{payload.get('version')} "
        f"(val_acc={payload.get('metrics', {}).get('val_acc', 0):.4f})"
    )
    return payload
