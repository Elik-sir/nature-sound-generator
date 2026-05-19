from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import torch

from src import config
from src.data.audio_io import load_waveform, resample_if_needed
from src.data.audio_transforms import WaveformParams, waveform_to_mel_128

logger = logging.getLogger(__name__)


def build_manifest_row(
    filename: str, target: int, fold: int, category: str, pt_path: Path
) -> dict:
    return {
        "filename": filename,
        "target": target,
        "fold": fold,
        "category": category,
        "path": pt_path.relative_to(config.PROJECT_ROOT).as_posix(),
    }


def process_one_file(
    wav_path: Path, out_path: Path, params: WaveformParams | None = None
) -> None:
    waveform, sr = load_waveform(wav_path)
    waveform = resample_if_needed(waveform, sr, config.SAMPLE_RATE)
    mel = waveform_to_mel_128(waveform, params)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mel, out_path)


def should_skip(wav_path: Path, out_path: Path) -> bool:
    if not out_path.exists():
        return False
    return out_path.stat().st_mtime >= wav_path.stat().st_mtime


def run(min_success_ratio: float = 0.9) -> list[dict]:
    config.require_dataset()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    total = 0
    ok = 0

    with open(config.META_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        total += 1
        filename = row["filename"]
        wav_path = config.RAW_AUDIO_DIR / filename
        stem = Path(filename).stem
        out_path = config.PROCESSED_DIR / f"{stem}.pt"
        try:
            if not wav_path.exists():
                logger.warning("Missing %s", wav_path)
                continue
            if not should_skip(wav_path, out_path):
                process_one_file(wav_path, out_path)
            ok += 1
            manifest.append(
                build_manifest_row(
                    filename,
                    int(row["target"]),
                    int(row["fold"]),
                    row["category"],
                    out_path,
                )
            )
        except Exception as e:
            logger.warning("Failed %s: %s", filename, e)

    if ok / max(total, 1) < min_success_ratio:
        raise RuntimeError(
            f"Processed only {ok}/{total} files; need >={min_success_ratio:.0%}"
        )

    with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote %d entries to %s", len(manifest), config.MANIFEST_PATH)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
