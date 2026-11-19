"""Helpers shared by the captioning engines."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch


def select_device() -> torch.device:
    """CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalise_map(grid: np.ndarray) -> List[List[float]]:
    """Scale one attention map to 0-1 (max = 1) and round for JSON."""
    grid = np.asarray(grid, dtype=np.float32)
    grid = grid / (grid.max() + 1e-8)
    return np.round(grid, 3).tolist()


def merge_wordpieces(
    pieces: Sequence[str], maps: Sequence[np.ndarray]
) -> Tuple[List[str], List[np.ndarray]]:
    """Join BERT word-pieces ("fr", "##is", "##bee") into words.

    The attention map of a merged word is the mean of its pieces' maps.
    """
    words: List[str] = []
    word_maps: List[List[np.ndarray]] = []
    for piece, grid in zip(pieces, maps):
        if piece.startswith("##") and words:
            words[-1] += piece[2:]
            word_maps[-1].append(grid)
        else:
            words.append(piece)
            word_maps.append([grid])
    return words, [np.mean(group, axis=0) for group in word_maps]


def block_repeated_ngrams(scores: torch.Tensor, seqs: List[List[int]], n: int, skip: int = 1) -> None:
    """Set -inf for every next token that would repeat an n-gram already in each sequence.

    ``skip`` leading tokens (e.g. the start token) are ignored.
    """
    if n <= 0:
        return
    for i, seq in enumerate(seqs):
        if len(seq) - skip < n - 1:
            continue
        prefix = tuple(seq[len(seq) - n + 1:]) if n > 1 else ()
        banned = {
            seq[j + n - 1]
            for j in range(skip, len(seq) - n + 1)
            if tuple(seq[j:j + n - 1]) == prefix
        }
        if banned:
            scores[i, list(banned)] = float("-inf")
