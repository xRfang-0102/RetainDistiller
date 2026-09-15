import copy
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class SpeechEncoder(nn.Module):
    def __init__(self, model, attention_mask="auto"):
        super().__init__()
        self.model = model
        self.attention_mask = attention_mask

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    def output_lengths(self, lengths):
        result = lengths.clone()
        for kernel, stride in zip(self.model.config.conv_kernel, self.model.config.conv_stride):
            result = torch.div(result - kernel, stride, rounding_mode="floor") + 1
        if (result < 1).any():
            raise ValueError("Waveform is shorter than the convolutional receptive field")
        return result

    def forward(self, waveforms, lengths, layer=-1, all_layers=False):
        raw_mask = torch.arange(waveforms.shape[1], device=lengths.device)[None] < lengths[:, None]
        use_mask = self.attention_mask
        if use_mask == "auto":
            use_mask = self.model.config.feat_extract_norm == "layer"
        outputs = self.model(
            waveforms,
            attention_mask=raw_mask.long() if use_mask else None,
            output_hidden_states=all_layers or layer != -1,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state if layer == -1 else outputs.hidden_states[layer]
        frame_lengths = self.output_lengths(lengths)
        mask = torch.arange(hidden.shape[1], device=lengths.device)[None] < frame_lengths[:, None]
        if all_layers:
            return outputs.hidden_states, mask, frame_lengths
        return hidden, mask, frame_lengths

    def freeze(self):
        self.requires_grad_(False)
        self.eval()
        return self


def load_encoder(path, expected_type=None, attention_mask="auto"):
    path = Path(path).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"Expected a local Hugging Face model directory: {path}")
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if config.model_type not in {"hubert", "wav2vec2", "wavlm"}:
        raise ValueError(f"Unsupported backbone: {config.model_type}")
    if expected_type and config.model_type != expected_type:
        raise ValueError(f"Expected {expected_type}, found {config.model_type}")
    if getattr(config, "add_adapter", False):
        raise ValueError("Temporal adapters are not supported by the matched-frame objective")
    config.apply_spec_augment = False
    config.layerdrop = 0.0
    model = AutoModel.from_pretrained(
        str(path), config=config, local_files_only=True, trust_remote_code=False,
        attn_implementation="eager",
    )
    return SpeechEncoder(model, attention_mask)


def build_student(teacher, config):
    student_config = copy.deepcopy(teacher.model.config)
    student_config.num_hidden_layers = config["num_layers"]
    student_config.layerdrop = 0.0
    student_config.apply_spec_augment = False
    for name in ("hidden_dropout", "attention_dropout", "activation_dropout", "feat_proj_dropout"):
        setattr(student_config, name, config["dropout"])
    model = AutoModel.from_config(student_config, attn_implementation="eager")
    initialization = config["initialization"]
    if initialization == "teacher":
        indices = config["teacher_init_layers"]
        if len(indices) != config["num_layers"]:
            raise ValueError("teacher_init_layers must contain one index per student layer")
        if any(i < 0 or i >= len(teacher.model.encoder.layers) for i in indices):
            raise ValueError("Invalid zero-based teacher initialization layer")
        source = teacher.model.state_dict()
        state = model.state_dict()
        for key in state:
            target_key = key
            if key.startswith("encoder.layers."):
                parts = key.split(".")
                parts[2] = str(indices[int(parts[2])])
                target_key = ".".join(parts)
            if target_key not in source or source[target_key].shape != state[key].shape:
                raise ValueError(f"Incompatible teacher initialization: {key}")
            state[key] = source[target_key].clone()
        model.load_state_dict(state, strict=True)
    elif initialization != "random":
        raise ValueError(f"Unknown initialization: {initialization}")
    if config["freeze_feature_encoder"]:
        model.feature_extractor.requires_grad_(False)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    return SpeechEncoder(model, teacher.attention_mask)
