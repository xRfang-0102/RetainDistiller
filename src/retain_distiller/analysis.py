import json
from pathlib import Path

import torch

from .data import AudioDataset, make_loader
from .engine import amp_context, restore_state, to_device
from .losses import posterior_kl
from .metrics import StreamingLinearCKA, retention
from .models.heads import LinguisticExtractor
from .system import TrainingSystem, probe_checkpoint, read_checkpoint


@torch.no_grad()
def analyze(checkpoint_path, manifest, output, device="cpu", precision="float32", batch_size=4, max_frames=None, probe_path=None):
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint["stage"] != "distill":
        raise ValueError("Knowledge analysis requires a distillation checkpoint")
    system = TrainingSystem(checkpoint["config"], "distill", initialize_probe=False)
    restore_state(system, checkpoint)
    system.to(device).eval()
    config = checkpoint["config"]
    if system.extractor is None or probe_path:
        probe = read_checkpoint(probe_path or probe_checkpoint(config))
        expected = system.metadata
        for key in ("backbone", "teacher_layer", "teacher_path", "hidden_size", "linguistic_size"):
            if probe["metadata"][key] != expected[key]:
                raise ValueError(f"Analysis probe metadata mismatch: {key}")
        extractor = LinguisticExtractor(system.teacher.hidden_size, config["probe"]["linguistic_size"], len(probe["metadata"]["symbols"]))
        extractor.load_state_dict({key.removeprefix("probe."): value for key, value in probe["model"].items()}, strict=True)
        extractor.to(device).eval()
    else:
        extractor = system.extractor
    dataset = AudioDataset(manifest, config["data"]["sample_rate"], config["data"]["normalize"])
    loader = make_loader(dataset, batch_size, 0, config["seed"])
    cka, divergence, frames = StreamingLinearCKA(), 0.0, 0
    temperature = config["method"]["temperature"]
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        with amp_context(torch.device(device), precision):
            ht, mask, _ = system.teacher(batch["waveforms"], batch["lengths"], config["backbone"]["teacher_layer"])
            hs, student_mask, _ = system.net["student"](batch["waveforms"], batch["lengths"])
            if not torch.equal(mask, student_mask):
                raise ValueError("Frame mask mismatch")
            zt, pt = extractor(ht)
            zs, ps = extractor(hs)
        teacher_z, student_z = zt[mask], zs[mask]
        teacher_p, student_p = pt[mask], ps[mask]
        if max_frames is not None:
            take = max_frames - frames
            teacher_z, student_z = teacher_z[:take], student_z[:take]
            teacher_p, student_p = teacher_p[:take], student_p[:take]
        n = len(teacher_z)
        valid = torch.ones(1, n, dtype=torch.bool, device=device)
        divergence += float(posterior_kl(student_p.unsqueeze(0), teacher_p.unsqueeze(0), valid, temperature, scale=False)) * n
        cka.update(teacher_z, student_z)
        frames += n
        if max_frames is not None and frames >= max_frames:
            break
    result = {
        "kl": divergence / frames, "temperature": temperature, "temperature_squared_scaled": False,
        "cka": cka.compute(), "frames": frames, "cka_estimator": "global_centered_linear_biased",
        "manifest": str(Path(manifest).resolve()), "checkpoint": str(Path(checkpoint_path).resolve()),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def retention_report(student_path, teacher_path, metric, output):
    student = json.loads(Path(student_path).read_text(encoding="utf-8"))
    teacher = json.loads(Path(teacher_path).read_text(encoding="utf-8"))
    if student.get("manifest") != teacher.get("manifest") or "manifest" not in student:
        raise ValueError("Retention requires evaluations on the same manifest")
    result = {"metric": metric, "student": student[metric], "teacher": teacher[metric], "retention_percent": retention(student[metric], teacher[metric], metric)}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
