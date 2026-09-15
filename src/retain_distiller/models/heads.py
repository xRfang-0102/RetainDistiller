import torch
from torch import nn


class LinguisticExtractor(nn.Module):
    def __init__(self, hidden_size, linguistic_size, vocab_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, linguistic_size), nn.GELU()
        )
        self.classifier = nn.Linear(linguistic_size, vocab_size)

    def forward(self, hidden):
        features = self.encoder(hidden)
        return features, self.classifier(features)


class LinguisticEnhancementHead(nn.Module):
    def __init__(self, hidden_size, linguistic_size, bottleneck, kernel_size):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("LEH kernel_size must be positive and odd")
        self.input = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, bottleneck), nn.GELU())
        self.temporal = nn.Conv1d(bottleneck, bottleneck, kernel_size, padding=kernel_size // 2, groups=bottleneck)
        self.output = nn.Linear(bottleneck, linguistic_size)

    def forward(self, hidden, mask):
        features = self.input(hidden) * mask.unsqueeze(-1)
        features = self.temporal(features.transpose(1, 2)).transpose(1, 2)
        return self.output(features) * mask.unsqueeze(-1)


class DownstreamHead(nn.Module):
    def __init__(self, hidden_size, output_size, kind, projection_size, dropout):
        super().__init__()
        if kind not in {"ctc", "classification"}:
            raise ValueError(f"Unsupported downstream kind: {kind}")
        self.kind = kind
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, projection_size), nn.GELU(), nn.Dropout(dropout)
        )
        self.output = nn.Linear(projection_size, output_size)

    def forward(self, hidden, mask):
        features = self.projection(hidden)
        if self.kind == "classification":
            features = (features * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.output(features)
