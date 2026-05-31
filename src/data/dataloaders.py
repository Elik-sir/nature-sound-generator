from __future__ import annotations

import json
import random
from pathlib import Path

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src import config
from src.data.audio_io import load_waveform, resample_if_needed
from src.data.audio_transforms import (
    waveform_to_log_mel_vae,
    waveform_to_mel_128,
    waveform_to_mel_sequence,
)
from src.data.augment import augment_mel, augment_waveform


def load_manifest(path: Path | None = None) -> list[dict]:
    manifest_path = path or config.MANIFEST_PATH
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def fold_split(
    records: list[dict],
    test_fold: int = config.TEST_FOLD,
    val_fold: int = config.VAL_FOLD,
) -> tuple[list[dict], list[dict], list[dict]]:
    test = [r for r in records if r["fold"] == test_fold]
    val = [r for r in records if r["fold"] == val_fold]
    train = [
        r for r in records if r["fold"] not in (test_fold, val_fold)
    ]
    return train, val, test


class ESC50Dataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        root: Path | None = None,
        augment: bool = False,
        vae_log_mel: bool = False,
        vae_crop_seconds: float | None = None,
    ):
        self.records = records
        self.root = root or config.PROJECT_ROOT
        self.augment = augment
        self.vae_log_mel = vae_log_mel
        self.vae_crop_seconds = vae_crop_seconds

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]
        spectrogram = self._load_spectrogram(record)
        if self.augment:
            spectrogram = augment_mel(spectrogram)
        return spectrogram, int(record["target"])

    def _load_spectrogram(self, record: dict) -> torch.Tensor:
        wav_path = self.root / "ESC-50-master" / "audio" / record["filename"]

        # VAE path: prefer raw waveform -> log-mel for better inversion quality.
        if self.vae_log_mel and wav_path.exists():
            waveform, sr = load_waveform(wav_path)
            waveform = resample_if_needed(waveform, sr, config.SAMPLE_RATE)
            if self.vae_crop_seconds and self.vae_crop_seconds > 0:
                crop_len = int(self.vae_crop_seconds * config.SAMPLE_RATE)
                if crop_len > 0:
                    if waveform.size(-1) >= crop_len:
                        start = random.randint(0, waveform.size(-1) - crop_len)
                        waveform = waveform[..., start : start + crop_len]
                    else:
                        repeats = (crop_len + waveform.size(-1) - 1) // waveform.size(-1)
                        waveform = waveform.repeat(1, repeats)[..., :crop_len]
            return waveform_to_log_mel_vae(waveform)

        # Optional waveform-level augmentation for train set.
        if (
            self.augment
            and config.ENABLE_AUDIO_AUGMENT
            and random.random() < config.AUDIO_AUG_PROB
        ):
            if wav_path.exists():
                waveform, sr = load_waveform(wav_path)
                waveform = resample_if_needed(waveform, sr, config.SAMPLE_RATE)
                waveform = augment_waveform(waveform, config.SAMPLE_RATE)
                return waveform_to_mel_128(waveform)

        # Fallback: use cached mel tensor.
        path = self.root / record["path"]
        spectrogram = torch.load(path, weights_only=True)
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
        return spectrogram


def _make_loader(
    records: list[dict],
    batch_size: int,
    shuffle: bool,
    *,
    augment: bool = False,
    vae_log_mel: bool = False,
    vae_crop_seconds: float | None = None,
) -> DataLoader:
    return DataLoader(
        ESC50Dataset(
            records,
            augment=augment,
            vae_log_mel=vae_log_mel,
            vae_crop_seconds=vae_crop_seconds,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.get_num_workers(),
    )


class LSTMSyntheticDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        root: Path | None = None,
        *,
        target_class: int = config.TARGET_CLASS,
        duration_s: int = config.LSTM_DURATION_S,
    ):
        self.records = records
        self.root = root or config.PROJECT_ROOT
        self.target_class = target_class
        self.duration_samples = int(duration_s * config.SAMPLE_RATE)
        self.min_target_segment_samples = int(0.5 * config.SAMPLE_RATE)
        self.target_pool = [r for r in records if int(r["target"]) == target_class]
        self.other_pool = [r for r in records if int(r["target"]) != target_class]
        if not self.target_pool:
            raise ValueError("LSTMSyntheticDataset requires at least one target-class record.")
        if not self.other_pool:
            raise ValueError("LSTMSyntheticDataset requires at least one non-target record.")

    def __len__(self) -> int:
        return len(self.records)

    def _load_record_waveform(self, record: dict) -> torch.Tensor:
        wav_path = self.root / "ESC-50-master" / "audio" / record["filename"]
        waveform, sr = load_waveform(wav_path)
        waveform = resample_if_needed(waveform, sr, config.SAMPLE_RATE)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    @staticmethod
    def _slice_or_repeat(waveform: torch.Tensor, length: int) -> torch.Tensor:
        wav = waveform
        if wav.size(-1) >= length:
            start = random.randint(0, wav.size(-1) - length)
            return wav[..., start : start + length]
        repeats = (length + wav.size(-1) - 1) // wav.size(-1)
        tiled = wav.repeat(1, repeats)
        return tiled[..., :length]

    def synthesize_example(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = idx  # samples are synthesized stochastically
        timeline = torch.zeros(1, self.duration_samples, dtype=torch.float32)
        sample_labels = torch.zeros(self.duration_samples, dtype=torch.float32)

        target_budget = int(0.4 * self.duration_samples)
        target_written = 0
        pos = 0

        while pos < self.duration_samples:
            remaining = self.duration_samples - pos
            if remaining <= 0:
                break
            remaining_target = max(target_budget - target_written, 0)
            if remaining_target >= remaining:
                use_target = True
            elif remaining_target == 0:
                use_target = False
            else:
                use_target = random.random() < (remaining_target / remaining)

            if use_target:
                max_seg = min(4 * config.SAMPLE_RATE, remaining)
                seg_len = random.randint(
                    min(self.min_target_segment_samples, max_seg),
                    max_seg,
                )
                seg_len = min(seg_len, remaining_target if remaining_target > 0 else seg_len)
                seg_len = max(1, min(seg_len, remaining))
                record = random.choice(self.target_pool)
                label_value = 1.0
            else:
                max_seg = min(4 * config.SAMPLE_RATE, remaining)
                seg_len = random.randint(1, max_seg)
                label_value = 0.0
                record = random.choice(self.other_pool) if random.random() < 0.5 else None

            if record is not None:
                segment = self._slice_or_repeat(self._load_record_waveform(record), seg_len)
                timeline[:, pos : pos + seg_len] = segment

            if label_value > 0:
                sample_labels[pos : pos + seg_len] = 1.0
                target_written += seg_len

            pos += seg_len

        mel = waveform_to_mel_sequence(timeline)  # [T, 128]
        frame_count = mel.shape[0]
        centers = (
            torch.arange(frame_count, dtype=torch.long) * config.HOP_LENGTH + config.N_FFT // 2
        ).clamp(max=self.duration_samples - 1)
        frame_labels = sample_labels[centers].float()
        return timeline, sample_labels, mel, frame_labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, mel, frame_labels = self.synthesize_example(idx)
        return mel, frame_labels


def collate_lstm(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    mels, labels = zip(*batch)
    mels_padded = pad_sequence(mels, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True)
    return mels_padded.float(), labels_padded.float()


def _make_lstm_loader(
    records: list[dict],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        LSTMSyntheticDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.get_num_workers(),
        collate_fn=collate_lstm,
    )


def _positive_segments(labels: torch.Tensor, sample_rate: int) -> list[dict[str, float]]:
    segments: list[dict[str, float]] = []
    active_start: int | None = None
    for i, value in enumerate(labels.tolist()):
        is_positive = value >= 0.5
        if is_positive and active_start is None:
            active_start = i
        elif not is_positive and active_start is not None:
            segments.append(
                {
                    "start_s": active_start / sample_rate,
                    "end_s": i / sample_rate,
                    "duration_s": (i - active_start) / sample_rate,
                }
            )
            active_start = None
    if active_start is not None:
        end = labels.numel()
        segments.append(
            {
                "start_s": active_start / sample_rate,
                "end_s": end / sample_rate,
                "duration_s": (end - active_start) / sample_rate,
            }
        )
    return segments


def export_lstm_debug_samples(
    records: list[dict],
    out_dir: Path | None = None,
    *,
    n_samples: int = 5,
    root: Path | None = None,
) -> list[Path]:
    """
    Save synthetic 30s LSTM examples for manual listening/debugging.
    Writes:
      - sample_XXX.wav
      - sample_XXX.labels.pt  (frame labels)
      - sample_XXX.json       (timing metadata with bird segments)
    """
    output_dir = out_dir or (config.PROCESSED_DIR / "lstm_debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = LSTMSyntheticDataset(records, root=root)

    written: list[Path] = []
    for i in range(n_samples):
        waveform, sample_labels, _, frame_labels = dataset.synthesize_example(i)
        wav_path = output_dir / f"sample_{i:03d}.wav"
        waveform_np = waveform.squeeze(0).detach().cpu().clamp(-1.0, 1.0).numpy()
        sf.write(str(wav_path), waveform_np, config.SAMPLE_RATE)

        labels_path = wav_path.with_suffix(".labels.pt")
        torch.save(frame_labels.cpu(), labels_path)

        meta = {
            "sample_rate": config.SAMPLE_RATE,
            "duration_s": waveform.shape[-1] / config.SAMPLE_RATE,
            "target_class": dataset.target_class,
            "bird_segments_seconds": _positive_segments(sample_labels, config.SAMPLE_RATE),
            "labels_path": labels_path.name,
        }
        meta_path = wav_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        written.append(wav_path)
    return written


def get_dataloaders(
    task: str,
    batch_size: int,
    test_fold: int = config.TEST_FOLD,
    val_fold: int = config.VAL_FOLD,
    *,
    vae_target_class: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if task not in ("cnn", "vae", "lstm"):
        raise ValueError(
            f"Unsupported task {task!r}. Expected 'cnn', 'vae', or 'lstm'."
        )

    records = load_manifest()
    if task == "vae":
        target = config.VAE_TARGET_CLASS if vae_target_class is None else vae_target_class
        records = [r for r in records if int(r["target"]) == int(target)]
        if not records:
            raise ValueError(
                f"No records found for VAE target class {target}. "
                "Check manifest and vae_target_class."
            )

    train_records, val_records, test_records = fold_split(
        records, test_fold=test_fold, val_fold=val_fold
    )

    if task == "lstm":
        train_loader = _make_lstm_loader(train_records, batch_size, shuffle=True)
        val_loader = _make_lstm_loader(val_records, batch_size, shuffle=False)
        test_loader = _make_lstm_loader(test_records, batch_size, shuffle=False)
    else:
        train_loader = _make_loader(
            train_records,
            batch_size,
            shuffle=True,
            augment=(task == "cnn"),
            vae_log_mel=(task == "vae"),
            vae_crop_seconds=(config.VAE_CROP_SECONDS if task == "vae" else None),
        )
        val_loader = _make_loader(
            val_records,
            batch_size,
            shuffle=False,
            vae_log_mel=(task == "vae"),
            vae_crop_seconds=None,
        )
        test_loader = _make_loader(
            test_records,
            batch_size,
            shuffle=False,
            vae_log_mel=(task == "vae"),
            vae_crop_seconds=None,
        )
    return train_loader, val_loader, test_loader
