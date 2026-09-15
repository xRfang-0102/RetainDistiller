import torch
from torch.nn import functional as F


def masked_mse(student, teacher, mask):
    if student.shape != teacher.shape or mask.shape != student.shape[:2]:
        raise ValueError("Matched representations must have identical batch, time, and feature dimensions")
    errors = (student.float() - teacher.detach().float()).square().mean(-1)
    return errors.masked_select(mask).mean()


def posterior_kl(student_logits, teacher_logits, mask, temperature=2.0, scale=True):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher = F.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    divergence = F.kl_div(student, teacher, reduction="none").sum(-1)
    return divergence.masked_select(mask).mean() * (temperature ** 2 if scale else 1.0)


def ctc_objective(logits, targets, frame_lengths, target_lengths, blank=0):
    if (target_lengths < 1).any():
        raise ValueError("CTC targets must not be empty")
    if ((targets < 0) | (targets >= logits.shape[-1]) | (targets == blank)).any():
        raise ValueError("CTC targets contain blank or invalid token indices")
    offset = 0
    for frames, size in zip(frame_lengths.tolist(), target_lengths.tolist()):
        sequence = targets[offset:offset + size]
        minimum = size + int((sequence[1:] == sequence[:-1]).sum())
        if minimum > frames:
            raise ValueError(f"Impossible CTC alignment: requires {minimum} frames, got {frames}")
        offset += size
    if offset != targets.numel():
        raise ValueError("Target lengths do not match concatenated targets")
    return F.ctc_loss(
        logits.float().log_softmax(-1).transpose(0, 1), targets,
        frame_lengths.cpu(), target_lengths.cpu(), blank=blank, reduction="mean", zero_infinity=False,
    )
