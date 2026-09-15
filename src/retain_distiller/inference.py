import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel

from .data import load_audio
from .models.backbone import load_encoder
from .system import read_checkpoint


def export_student(checkpoint_path, output_dir):
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint["stage"] != "distill":
        raise ValueError("Only distillation checkpoints contain an exportable student")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {output}")
    raw_config = dict(checkpoint["student_config"])
    model_type = raw_config.pop("model_type")
    model = AutoModel.from_config(AutoConfig.for_model(model_type, **raw_config), attn_implementation="eager")
    prefix = "student.model."
    state = {key.removeprefix(prefix): value for key, value in checkpoint["model"].items() if key.startswith(prefix)}
    model.load_state_dict(state, strict=True)
    model.eval().save_pretrained(str(output), safe_serialization=True)
    config = checkpoint["config"]
    metadata = {
        "format_version": 1, "sample_rate": config["data"]["sample_rate"],
        "normalize": config["data"]["normalize"], "attention_mask": config["backbone"]["attention_mask"],
        "training_step": checkpoint["step"], "method": config["method"]["name"],
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "retain_distiller.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"export": str(output), **metadata}))
    return output


class StudentInference:
    def __init__(self, model_dir, device="cpu"):
        path = Path(model_dir) / "retain_distiller.json"
        self.metadata = json.loads(path.read_text(encoding="utf-8"))
        self.device = torch.device(device)
        self.encoder = load_encoder(model_dir, attention_mask=self.metadata["attention_mask"]).to(device).eval()

    @torch.no_grad()
    def encode_file(self, path):
        waveform = load_audio(path, self.metadata["sample_rate"], self.metadata["normalize"]).to(self.device)
        lengths = torch.tensor([len(waveform)], device=self.device)
        features, _, frame_lengths = self.encoder(waveform.unsqueeze(0), lengths)
        return features[0, :int(frame_lengths[0])].cpu()


def infer(model_dir, audio_path, output, device="cpu"):
    features = StudentInference(model_dir, device).encode_file(audio_path)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "audio": str(Path(audio_path).resolve())}, output)
    print(json.dumps({"output": str(output), "shape": list(features.shape)}))
