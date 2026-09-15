import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler


def read_jsonl(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate utterance IDs in {path}")
    for row in rows:
        audio_path = Path(row["audio"]).expanduser()
        row["audio"] = str((path.parent / audio_path).resolve()) if not audio_path.is_absolute() else str(audio_path)
    return rows


def load_symbols(path, blank=False):
    with Path(path).open(encoding="utf-8") as stream:
        symbols = json.load(stream)
    if not isinstance(symbols, list) or not symbols or any(not isinstance(x, str) for x in symbols):
        raise ValueError("Vocabulary must be a nonempty JSON list of strings")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Vocabulary contains duplicate symbols")
    if blank and symbols[0] != "<blank>":
        raise ValueError("CTC blank must be the first vocabulary entry")
    return symbols


def load_audio(path, sample_rate=16000, normalize=True):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, rate // divisor).astype(np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    waveform = torch.from_numpy(audio.copy())
    if normalize:
        waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var(unbiased=False) + 1e-7)
    return waveform


class AudioDataset(Dataset):
    def __init__(self, manifest, sample_rate=16000, normalize=True, target=None, symbols=None, crop_seconds=None, seed=42):
        self.rows = read_jsonl(manifest)
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.target = target
        self.symbols = {value: i for i, value in enumerate(symbols or [])}
        self.crop_samples = int(crop_seconds * sample_rate) if crop_seconds else None
        self.seed = seed
        self.epoch = 0
        if self.crop_samples and target is not None:
            raise ValueError("Cropping labeled utterances is disabled to preserve target alignment")
        if target:
            for row in self.rows:
                self.encode_target(row)

    def encode_target(self, row):
        if self.target == "label":
            return self.symbols[str(row["label"])]
        if self.target == "tokens":
            sequence = row["tokens"]
        elif self.target == "text":
            sequence = list("|".join(row["text"].upper().split()))
        else:
            return None
        if not sequence:
            raise ValueError(f"Empty target in {row['id']}")
        try:
            result = [self.symbols[token] for token in sequence]
        except KeyError as error:
            raise ValueError(f"Unknown target symbol in {row['id']}: {error}") from error
        if 0 in result:
            raise ValueError("CTC targets must not contain blank")
        return torch.tensor(result, dtype=torch.long)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        waveform = load_audio(row["audio"], self.sample_rate, normalize=False)
        if self.crop_samples and len(waveform) > self.crop_samples:
            rng = random.Random(self.seed + self.epoch * len(self) + index)
            start = rng.randint(0, len(waveform) - self.crop_samples)
            waveform = waveform[start:start + self.crop_samples]
        if self.normalize:
            waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var(unbiased=False) + 1e-7)
        result = {"id": row["id"], "waveform": waveform}
        if self.target is not None:
            result["target"] = self.encode_target(row)
        return result


def collate_audio(items):
    batch = {
        "ids": [item["id"] for item in items],
        "waveforms": pad_sequence([item["waveform"] for item in items], batch_first=True),
        "lengths": torch.tensor([len(item["waveform"]) for item in items], dtype=torch.long),
    }
    if "target" in items[0]:
        if isinstance(items[0]["target"], int):
            batch["labels"] = torch.tensor([item["target"] for item in items], dtype=torch.long)
        else:
            targets = [item["target"] for item in items]
            batch["targets"] = torch.cat(targets)
            batch["target_lengths"] = torch.tensor([len(target) for target in targets], dtype=torch.long)
    return batch


class EpochBatches(Sampler):
    def __init__(self, size, batch_size, seed, epoch=0, start=0):
        if size < batch_size:
            raise ValueError("Training manifest must contain at least one complete microbatch")
        self.size, self.batch_size, self.seed, self.epoch, self.start = size, batch_size, seed, epoch, start

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        batches = [indices[i:i + self.batch_size] for i in range(0, self.size - self.batch_size + 1, self.batch_size)]
        yield from batches[self.start:]

    def __len__(self):
        return max(0, self.size // self.batch_size - self.start)


def make_loader(dataset, batch_size, workers, seed, epoch=None, start=0):
    generator = torch.Generator().manual_seed(seed + (epoch or 0))
    common = dict(num_workers=workers, collate_fn=collate_audio, pin_memory=torch.cuda.is_available(), generator=generator)
    if epoch is not None:
        dataset.epoch = epoch
        return DataLoader(dataset, batch_sampler=EpochBatches(len(dataset), batch_size, seed, epoch, start), **common)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **common)


def validate_splits(train_dataset, valid_dataset):
    train_ids = {row["id"] for row in train_dataset.rows}
    valid_ids = {row["id"] for row in valid_dataset.rows}
    train_paths = {str(Path(row["audio"]).resolve()) for row in train_dataset.rows}
    valid_paths = {str(Path(row["audio"]).resolve()) for row in valid_dataset.rows}
    if train_ids & valid_ids or train_paths & valid_paths:
        raise ValueError("Training and validation manifests overlap")
