from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .data import AudioDataset, load_symbols
from .losses import ctc_objective, masked_mse, posterior_kl
from .models import LinguisticEnhancementHead, LinguisticExtractor, build_student, load_encoder
from .models.heads import DownstreamHead


def run_directory(config, stage):
    if config["paths"]["output_dir"]:
        return Path(config["paths"]["output_dir"])
    suffix = "probe" if stage == "probe" else config["method"]["name"]
    if stage == "downstream":
        suffix = f"{suffix}/{config['downstream']['name']}"
    return Path(config["paths"]["runs"]) / config["backbone"]["name"] / suffix


def probe_checkpoint(config):
    if config["paths"]["probe_checkpoint"]:
        return Path(config["paths"]["probe_checkpoint"])
    return Path(config["paths"]["runs"]) / config["backbone"]["name"] / "probe" / "best.pt"


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return checkpoint


class TrainingSystem(nn.Module):
    def __init__(self, config, stage, initialize_probe=True):
        super().__init__()
        self.config, self.stage = config, stage
        self.net = nn.ModuleDict()
        self.teacher = None
        self.extractor = None
        self.symbols = None
        self.target = None
        self.kind = None
        self.unit = None
        self.metadata = {"stage": stage}
        backbone, probe = config["backbone"], config["probe"]
        if stage in {"probe", "distill"}:
            self.teacher = load_encoder(backbone["teacher_path"], backbone["name"], backbone["attention_mask"]).freeze()
            layer = backbone["teacher_layer"]
            if layer < 1 or layer > self.teacher.model.config.num_hidden_layers:
                raise ValueError("teacher_layer is a one-based Transformer layer index")
            self.metadata.update({
                "backbone": backbone["name"], "teacher_layer": layer,
                "teacher_path": str(Path(backbone["teacher_path"]).expanduser().resolve()),
                "hidden_size": self.teacher.hidden_size, "linguistic_size": probe["linguistic_size"],
            })
        if stage == "probe":
            self.symbols = load_symbols(config["data"]["phoneme_vocab"], blank=True)
            self.target, self.kind, self.unit = "tokens", "ctc", "phone"
            self.net["probe"] = LinguisticExtractor(self.teacher.hidden_size, probe["linguistic_size"], len(self.symbols))
        elif stage == "distill":
            method = config["method"]["objective"]
            if method not in {"mse", "posterior", "feature", "direct_ctc"}:
                raise ValueError(f"Unknown distillation objective: {method}")
            self.net["student"] = build_student(self.teacher, config["student"])
            if method in {"posterior", "feature"} or config["leh"]["enabled"]:
                self.symbols = load_symbols(config["data"]["phoneme_vocab"], blank=True)
                self.extractor = LinguisticExtractor(self.teacher.hidden_size, probe["linguistic_size"], len(self.symbols))
                if initialize_probe:
                    checkpoint = read_checkpoint(probe_checkpoint(config))
                    expected = {**self.metadata, "stage": "probe", "symbols": self.symbols}
                    if checkpoint["metadata"] != expected:
                        raise ValueError("Probe metadata does not match teacher, selected layer, dimensions, or vocabulary")
                    self.extractor.load_state_dict({key.removeprefix("probe."): value for key, value in checkpoint["model"].items()}, strict=True)
                self.extractor.requires_grad_(False).eval()
            if method == "direct_ctc":
                self.symbols = load_symbols(config["data"]["phoneme_vocab"], blank=True)
                self.target = "tokens"
                self.net["direct_head"] = LinguisticExtractor(self.teacher.hidden_size, probe["linguistic_size"], len(self.symbols))
            if config["leh"]["enabled"]:
                leh = config["leh"]
                self.net["leh"] = LinguisticEnhancementHead(self.teacher.hidden_size, probe["linguistic_size"], leh["bottleneck"], leh["kernel_size"])
        elif stage == "downstream":
            downstream = config["downstream"]
            self.net["upstream"] = load_encoder(downstream["upstream_path"], backbone["name"], backbone["attention_mask"])
            self.net["upstream"].requires_grad_(not downstream["freeze_upstream"])
            self.kind = downstream["kind"]
            self.target = downstream["target"]
            self.unit = downstream["unit"]
            self.symbols = load_symbols(downstream["vocab"], blank=self.kind == "ctc")
            self.net["head"] = DownstreamHead(self.net["upstream"].hidden_size, len(self.symbols), self.kind, downstream["projection_size"], downstream["dropout"])
            self.metadata.update({"task": downstream["name"], "kind": self.kind, "unit": self.unit})
        else:
            raise ValueError(f"Unknown training stage: {stage}")
        if self.symbols is not None:
            self.metadata["symbols"] = self.symbols

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        if self.extractor is not None:
            self.extractor.eval()
        if self.stage == "downstream" and self.config["downstream"]["freeze_upstream"]:
            self.net["upstream"].eval()
        return self

    def dataset(self, split, manifest=None):
        data = self.config["data"]
        if self.stage == "downstream":
            manifest = manifest or self.config["downstream"][f"{split}_manifest"]
        else:
            manifest = manifest or data[f"{split}_manifest"]
        crop = data["distill_crop_seconds"] if self.stage == "distill" and self.target is None and split == "train" else None
        return AudioDataset(manifest, data["sample_rate"], data["normalize"], self.target, self.symbols, crop, self.config["seed"])

    def forward(self, batch):
        waveforms, lengths = batch["waveforms"], batch["lengths"]
        if self.stage == "downstream":
            with torch.set_grad_enabled(torch.is_grad_enabled() and not self.config["downstream"]["freeze_upstream"]):
                hidden, mask, frame_lengths = self.net["upstream"](waveforms, lengths, self.config["downstream"]["layer"])
            logits = self.net["head"](hidden, mask)
        else:
            with torch.no_grad():
                teacher_hidden, mask, frame_lengths = self.teacher(waveforms, lengths, self.config["backbone"]["teacher_layer"])
            if self.stage == "probe":
                _, logits = self.net["probe"](teacher_hidden)
            else:
                student_hidden, student_mask, student_lengths = self.net["student"](waveforms, lengths)
                if not torch.equal(mask, student_mask) or not torch.equal(frame_lengths, student_lengths):
                    raise ValueError("Teacher and student frame grids do not match")
                rep = masked_mse(student_hidden, teacher_hidden, mask)
                losses = {"rep": rep}
                loss = rep
                objective = self.config["method"]["objective"]
                if self.extractor is not None:
                    with torch.no_grad():
                        teacher_features, teacher_logits = self.extractor(teacher_hidden)
                    student_features, student_logits = self.extractor(student_hidden)
                    if objective == "posterior":
                        losses["kp"] = posterior_kl(student_logits, teacher_logits, mask, self.config["method"]["temperature"])
                        loss = loss + self.config["method"]["lambda_kp"] * losses["kp"]
                    elif objective == "feature":
                        losses["feature"] = masked_mse(student_features, teacher_features, mask)
                        loss = loss + self.config["method"]["lambda_kp"] * losses["feature"]
                    if "leh" in self.net:
                        losses["enh"] = masked_mse(self.net["leh"](student_hidden, mask), teacher_features, mask)
                        loss = loss + self.config["method"]["beta"] * losses["enh"]
                if objective == "direct_ctc":
                    _, logits = self.net["direct_head"](student_hidden)
                    losses["ctc"] = ctc_objective(logits, batch["targets"], frame_lengths, batch["target_lengths"])
                    loss = loss + self.config["method"]["lambda_ctc"] * losses["ctc"]
                return {"loss": loss, **losses}
        if self.kind == "ctc":
            loss = ctc_objective(logits, batch["targets"], frame_lengths, batch["target_lengths"])
        else:
            loss = F.cross_entropy(logits.float(), batch["labels"])
        return {"loss": loss, "logits": logits, "frame_lengths": frame_lengths}
