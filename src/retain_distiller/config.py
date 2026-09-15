import copy
from pathlib import Path

import yaml


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_config(path, visited=()):
    path = Path(path).resolve()
    if path in visited:
        raise ValueError(f"Circular config inheritance: {path}")
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    parents = values.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    result = {}
    for parent in parents:
        result = merge(result, read_config(path.parent / parent, (*visited, path)))
    return merge(result, values)


def load_config(path, overlays=(), overrides=()):
    result = read_config(path)
    for overlay in overlays:
        result = merge(result, read_config(overlay))
    for override in overrides:
        key, value = override.split("=", 1)
        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise KeyError(f"Unknown config path: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    return result
