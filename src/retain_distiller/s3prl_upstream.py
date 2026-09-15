import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .models.backbone import load_encoder


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt, **kwargs):
        super().__init__()
        metadata = json.loads((Path(ckpt) / "retain_distiller.json").read_text(encoding="utf-8"))
        self.encoder = load_encoder(ckpt, attention_mask=metadata["attention_mask"])
        self.normalize = metadata["normalize"]

    def get_downsample_rates(self, key):
        return math.prod(self.encoder.model.config.conv_stride)

    def forward(self, wavs):
        if self.normalize:
            wavs = [(wave - wave.mean()) / torch.sqrt(wave.var(unbiased=False) + 1e-7) for wave in wavs]
        lengths = torch.tensor([len(wave) for wave in wavs], device=wavs[0].device)
        hidden, _, frame_lengths = self.encoder(pad_sequence(wavs, batch_first=True), lengths, all_layers=True)
        return {"hidden_states": list(hidden), "last_hidden_state": hidden[-1], "lengths": frame_lengths}
