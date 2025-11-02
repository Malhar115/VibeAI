import os
import random
from typing import List, Tuple, Dict

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def list_audio_files_with_labels(root_dir: str) -> List[Tuple[str, str]]:
    """
    Assumes structure:
      root_dir/
        happy/*.wav
        sad/*.wav
        ...
    Returns list of (file_path, label).
    """
    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(root_dir):
        return pairs
    for label in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, label)
        if not os.path.isdir(full):
            continue
        for fname in sorted(os.listdir(full)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in AUDIO_EXTS:
                pairs.append((os.path.join(full, fname), label))
    return pairs


def stratified_split(
    pairs: List[Tuple[str, str]],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    by_label: Dict[str, List[Tuple[str, str]]] = {}
    for p, y in pairs:
        by_label.setdefault(y, []).append((p, y))
    rng = random.Random(seed)
    train: List[Tuple[str, str]] = []
    test: List[Tuple[str, str]] = []
    for y, items in by_label.items():
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * test_size))) if len(items) > 1 else 1
        test.extend(items[:n_test])
        train.extend(items[n_test:] if len(items) > n_test else [])
    if len(train) == 0 and len(test) > 0:
        train.append(test.pop())
    return train, test


def build_label_maps(labels: List[str]):
    uniq = sorted(set(labels))
    lab2idx = {l: i for i, l in enumerate(uniq)}
    idx2lab = {i: l for l, i in lab2idx.items()}
    return lab2idx, idx2lab
