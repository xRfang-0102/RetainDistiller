import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import make_loader, validate_splits
from .metrics import ErrorRate, greedy_ctc, split_targets
from .system import TrainingSystem, read_checkpoint, run_directory


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def amp_context(device, precision):
    if precision == "float32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("Set runtime.precision=float32 for CPU execution")
    if precision not in {"float16", "bfloat16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; choose float16 or float32")
    return torch.autocast("cuda", dtype=getattr(torch, precision))


def learning_rate_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def training_options(config, stage):
    return config[{"probe": "probe_train", "distill": "train", "downstream": "downstream_train"}[stage]]


@torch.no_grad()
def evaluate(system, loader, device, precision, max_batches=None):
    system.eval()
    total, count, correct = {}, 0, 0
    errors = ErrorRate(system.symbols, system.unit) if system.kind == "ctc" else None
    for i, raw_batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = to_device(raw_batch, device)
        with amp_context(device, precision):
            result = system(batch)
        size = len(batch["ids"])
        for key, value in result.items():
            if value.ndim == 0:
                if not torch.isfinite(value):
                    raise FloatingPointError(f"Nonfinite validation {key}")
                total[key] = total.get(key, 0.0) + float(value) * size
        if errors is not None:
            errors.update(greedy_ctc(result["logits"], result["frame_lengths"]), split_targets(batch["targets"], batch["target_lengths"]))
        elif system.kind == "classification":
            correct += int((result["logits"].argmax(-1) == batch["labels"]).sum())
        count += size
    if not count:
        raise ValueError("Evaluation loader is empty")
    metrics = {key: value / count for key, value in total.items()}
    metrics["utterances"] = count
    if errors is not None:
        metrics["wer" if system.unit == "word" else "per"] = errors.compute()
    elif system.kind == "classification":
        metrics["accuracy"] = correct / count
    return metrics


def save_training_state(path, system, optimizer, scheduler, scaler, step, epoch, batch_offset, best):
    numpy_rng = np.random.get_state()
    payload = {
        "format_version": 1, "stage": system.stage, "config": system.config,
        "metadata": system.metadata, "model": system.net.state_dict(),
        "student_config": system.net["student"].model.config.to_dict() if system.stage == "distill" else None,
        "extractor": system.extractor.state_dict() if system.extractor is not None else None,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "step": step, "epoch": epoch, "batch_offset": batch_offset, "best": best,
        "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), numpy_rng[2], numpy_rng[3], numpy_rng[4]],
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    atomic_save(path, payload)


def restore_state(system, checkpoint):
    if checkpoint["stage"] != system.stage or checkpoint["metadata"] != system.metadata:
        raise ValueError("Checkpoint stage or metadata does not match")
    system.net.load_state_dict(checkpoint["model"], strict=True)
    if system.extractor is not None:
        system.extractor.load_state_dict(checkpoint["extractor"], strict=True)


def train(config, stage, resume=None):
    seed_everything(config["seed"])
    options = training_options(config, stage)
    total_steps, accumulation = options["max_steps"], options["accumulation_steps"]
    if total_steps < 1 or accumulation < 1 or options["batch_size"] < 1:
        raise ValueError("Steps, accumulation, and batch size must be positive")
    if not 0 <= options["warmup_steps"] < total_steps:
        raise ValueError("warmup_steps must be nonnegative and smaller than max_steps")
    device = torch.device(config["runtime"]["device"])
    precision = config["runtime"]["precision"]
    amp_context(device, precision)
    output = run_directory(config, stage)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError(f"Run already exists; use --resume or change paths.output_dir: {output}")
    system = TrainingSystem(config, stage, initialize_probe=resume is None).to(device)
    parameters = [parameter for parameter in system.net.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=options["learning_rate"], betas=tuple(options["betas"]), weight_decay=options["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, options["warmup_steps"], total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "float16")
    step = epoch = batch_offset = 0
    best = float("inf")
    if resume:
        checkpoint = read_checkpoint(resume)
        if checkpoint["config"] != config:
            raise ValueError("Resume requires the same resolved config; use a new run for changed settings")
        restore_state(system, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        step, epoch, batch_offset, best = (checkpoint[key] for key in ("step", "epoch", "batch_offset", "best"))
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        state = checkpoint["numpy_rng"]
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    train_data, valid_data = system.dataset("train"), system.dataset("valid")
    validate_splits(train_data, valid_data)
    workers = config["runtime"]["num_workers"]
    valid_loader = make_loader(valid_data, options["eval_batch_size"], workers, config["seed"])
    iterator = iter(make_loader(train_data, options["batch_size"], workers, config["seed"], epoch, batch_offset))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"stage": stage, "output": str(output), "trainable_parameters": sum(p.numel() for p in parameters), "effective_batch_size": options["batch_size"] * accumulation}), flush=True)
    while step < total_steps:
        system.train()
        optimizer.zero_grad(set_to_none=True)
        values = {}
        for _ in range(accumulation):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                epoch, batch_offset = epoch + 1, 0
                iterator = iter(make_loader(train_data, options["batch_size"], workers, config["seed"], epoch))
                raw_batch = next(iterator)
            batch_offset += 1
            batch = to_device(raw_batch, device)
            with amp_context(device, precision):
                result = system(batch)
                loss = result["loss"] / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at update {step}")
            scaler.scale(loss).backward()
            for key, value in result.items():
                if value.ndim == 0:
                    values[key] = values.get(key, 0.0) + float(value.detach()) / accumulation
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, options["max_grad_norm"])
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at update {step}; try bfloat16 or float32")
        used_lr = optimizer.param_groups[0]["lr"]
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        if step % options["log_every"] == 0 or step == 1:
            record = {"step": step, "split": "train", "lr": used_lr, **values}
            print(json.dumps(record), flush=True)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        improved = False
        if step % options["eval_every"] == 0 or step == total_steps:
            metrics = evaluate(system, valid_loader, device, precision, options["eval_max_batches"])
            score = metrics.get("per", metrics.get("wer", -metrics["accuracy"] if "accuracy" in metrics else metrics["loss"]))
            improved = score < best
            best = min(best, score)
            record = {"step": step, "split": "valid", **metrics}
            print(json.dumps(record), flush=True)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        if improved:
            save_training_state(output / "best.pt", system, optimizer, scheduler, scaler, step, epoch, batch_offset, best)
        if improved or step % options["save_every"] == 0 or step == total_steps:
            save_training_state(output / "last.pt", system, optimizer, scheduler, scaler, step, epoch, batch_offset, best)
    return str(output)


def evaluate_checkpoint(path, manifest, device="cpu", precision="float32", batch_size=4, output=None):
    checkpoint = read_checkpoint(path)
    system = TrainingSystem(checkpoint["config"], checkpoint["stage"], initialize_probe=False)
    restore_state(system, checkpoint)
    system.to(device)
    dataset = system.dataset("test", manifest)
    loader = make_loader(dataset, batch_size, 0, checkpoint["config"]["seed"])
    metrics = evaluate(system, loader, torch.device(device), precision)
    metrics.update({"checkpoint": str(Path(path).resolve()), "manifest": str(Path(manifest).resolve()), "stage": checkpoint["stage"]})
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics
