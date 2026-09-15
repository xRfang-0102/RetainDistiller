import torch


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def greedy_ctc(logits, lengths, blank=0):
    predictions = logits.argmax(-1).cpu()
    results = []
    for sequence, length in zip(predictions, lengths.tolist()):
        collapsed = torch.unique_consecutive(sequence[:length]).tolist()
        results.append([token for token in collapsed if token != blank])
    return results


def split_targets(targets, lengths):
    return [part.tolist() for part in targets.cpu().split(lengths.cpu().tolist())]


class ErrorRate:
    def __init__(self, symbols, unit="phone"):
        self.symbols, self.unit = symbols, unit
        self.errors = 0
        self.total = 0

    def decode(self, ids):
        tokens = [self.symbols[index] for index in ids]
        if self.unit == "word":
            return "".join(tokens).replace("|", " ").split()
        return tokens

    def update(self, predictions, references):
        for prediction, reference in zip(predictions, references):
            expected, actual = self.decode(reference), self.decode(prediction)
            self.errors += edit_distance(expected, actual)
            self.total += len(expected)

    def compute(self):
        if not self.total:
            raise ValueError("No reference units available for error rate")
        return self.errors / self.total


def retention(student, teacher, metric):
    if metric == "per":
        student, teacher = 1.0 - student, 1.0 - teacher
    elif metric != "accuracy":
        raise ValueError("Retention supports per or accuracy, both supplied as fractions")
    if teacher <= 0:
        raise ValueError("Teacher performance must be positive")
    return 100.0 * student / teacher


class StreamingLinearCKA:
    def __init__(self):
        self.n = 0
        self.sx = self.sy = self.xx = self.yy = self.xy = None

    def update(self, x, y):
        x, y = x.detach().double().cpu(), y.detach().double().cpu()
        if len(x) != len(y) or x.ndim != 2 or y.ndim != 2:
            raise ValueError("CKA requires matched two-dimensional observations")
        if not len(x):
            return
        values = (x.sum(0), y.sum(0), x.T @ x, y.T @ y, x.T @ y)
        if self.n == 0:
            self.sx, self.sy, self.xx, self.yy, self.xy = values
        else:
            for key, value in zip(("sx", "sy", "xx", "yy", "xy"), values):
                setattr(self, key, getattr(self, key) + value)
        self.n += len(x)

    def compute(self):
        if self.n < 2:
            raise ValueError("CKA requires at least two observations")
        xx = self.xx - torch.outer(self.sx, self.sx) / self.n
        yy = self.yy - torch.outer(self.sy, self.sy) / self.n
        xy = self.xy - torch.outer(self.sx, self.sy) / self.n
        denominator = torch.linalg.matrix_norm(xx) * torch.linalg.matrix_norm(yy)
        if denominator <= 0:
            raise ValueError("CKA is undefined for constant representations")
        return float(xy.square().sum() / denominator)
