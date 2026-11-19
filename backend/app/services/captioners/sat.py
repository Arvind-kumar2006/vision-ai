"""
SATCaptioner: pretrained Show, Attend and Tell (ResNet-101, COCO 2014).

Model code : model/captioning/sat_models.py
Weights    : weights/sat_resnet101_coco.pth
             (converted once by `python -m model.captioning.convert_checkpoint`)

Per request:
  1. Resize to 256x256, ImageNet mean/std normalisation.
  2. Encoder (ResNet-101 + adaptive pool) -> (1, 14, 14, 2048).
  3. Beam search (default width 3) with the gated soft-attention LSTM decoder.
  4. Return caption, words and one 14x14 attention map per word (scaled 0-1).
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from ...config import BACKEND_ROOT, settings

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from model.captioning.sat_models import DecoderWithAttention, Encoder  # noqa: E402

log = logging.getLogger("backend.caption")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SPECIAL_TOKENS = {"<start>", "<end>", "<pad>", "<unk>"}


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_inference_cfg() -> dict:
    try:
        with open(settings.config_path, "r") as f:
            return (yaml.safe_load(f) or {}).get("inference", {}) or {}
    except FileNotFoundError:
        return {}


class SATCaptioner:
    """Loads the converted COCO model once and serves beam-search captions."""

    name = "sat"
    label = "Show, Attend and Tell (ResNet-101 + LSTM, COCO 2014)"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = dict(cfg if cfg is not None else _load_inference_cfg())
        cfg.update(cfg.get("sat") or {})  # model-specific keys override shared ones
        weights = Path(cfg.get("weights", "weights/sat_resnet101_coco.pth"))
        self.checkpoint_path = weights if weights.is_absolute() else BACKEND_ROOT / weights
        self.image_size = int(cfg.get("image_size", 256))
        self.default_beam_size = int(cfg.get("beam_size", 5))
        self.max_caption_len = int(cfg.get("max_caption_len", 50))
        # Forbid repeating any word n-gram within a caption (0 = off). Stops loops
        # like "a dog sitting in the grass next to a dog".
        self.no_repeat_ngram = int(cfg.get("no_repeat_ngram", 2))

        self.device = _select_device()
        self.lock = threading.Lock()
        self.ready = False
        self.error: str | None = None
        self.vocab_size = 0
        self.meta: dict = {}

        self.encoder: Encoder | None = None
        self.decoder: DecoderWithAttention | None = None
        self.word_map: dict[str, int] = {}
        self.rev_word_map: dict[int, str] = {}

        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        path = self.checkpoint_path
        if not path.exists():
            self.error = (f"Weights not found: {path}. "
                          "Run `python -m model.captioning.convert_checkpoint` from backend/.")
            log.warning(self.error)
            return
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            meta = ckpt.get("meta", {})
            word_map = ckpt["word_map"]

            encoder = Encoder(encoded_image_size=int(meta.get("encoded_image_size", 14)))
            decoder = DecoderWithAttention(
                attention_dim=int(meta.get("attention_dim", 512)),
                embed_dim=int(meta.get("embed_dim", 512)),
                decoder_dim=int(meta.get("decoder_dim", 512)),
                vocab_size=len(word_map),
                encoder_dim=int(meta.get("encoder_dim", 2048)),
            )
            encoder.load_state_dict(ckpt["encoder_state"], strict=True)
            decoder.load_state_dict(ckpt["decoder_state"], strict=True)

            self.encoder = encoder.to(self.device).eval()
            self.decoder = decoder.to(self.device).eval()
            self.word_map = word_map
            self.rev_word_map = {v: k for k, v in word_map.items()}
            self.vocab_size = len(word_map)
            self.meta = meta
            self.ready = True

            # Warm-up pass so the first real request doesn't pay kernel setup.
            # Uses the default beam width: GPU (MPS/CUDA) kernels are shape-specific,
            # so a beam-1 warm-up does not warm the beam-3 path.
            started = time.perf_counter()
            noise = np.random.default_rng(0).integers(0, 256, (self.image_size, self.image_size, 3),
                                                       dtype=np.uint8)
            self.caption(Image.fromarray(noise), beam_size=self.default_beam_size, max_len=12)
            log.info("Caption model warm-up done in %.0f ms",
                     (time.perf_counter() - started) * 1000)
            log.info("SAT caption model loaded: vocab %d on %s (epoch %s, val BLEU-4 %s)",
                     self.vocab_size, self.device, meta.get("epoch"), meta.get("val_bleu4"))
        except Exception as exc:
            self.error = f"Failed to load caption model: {exc}"
            log.exception(self.error)

    # ------------------------------------------------------------------
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        # reducing_gap: shrink in cheap integer steps first, then bicubic, so very large
        # photos (e.g. 14000x9000) resize in milliseconds instead of ~1.5 s.
        img = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC,
                                          reducing_gap=3.0)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def caption(self, image: Image.Image, beam_size: int | None = None,
                max_len: int | None = None) -> dict:
        """Caption a PIL image. Returns caption, words, per-word alphas (0-1), timing."""
        if not self.ready:
            raise RuntimeError(self.error or "Caption model is not loaded.")

        started = time.perf_counter()
        beam = int(beam_size or self.default_beam_size)
        limit = int(max_len or self.max_caption_len)
        tensor = self._preprocess(image)

        with self.lock:
            seq, alphas = self._beam_search(tensor, beam, limit)

        words: list[str] = []
        word_alphas: list[list[list[float]]] = []
        for t in range(1, len(seq)):  # index 0 is <start>
            word = self.rev_word_map.get(seq[t], "<unk>")
            if word in ("<end>", "<pad>"):
                break
            if word in SPECIAL_TOKENS:
                continue  # drop <unk> (and any stray <start>) together with its map
            a = np.asarray(alphas[t], dtype=np.float32)
            a = a / (a.max() + 1e-8)
            words.append(word)
            word_alphas.append(np.round(a, 3).tolist())

        return {
            "caption": " ".join(words),
            "words": words,
            "alphas": word_alphas,
            "grid_size": len(word_alphas[0]) if word_alphas else 0,
            "grid_shape": [len(word_alphas[0]), len(word_alphas[0][0])] if word_alphas else [0, 0],
            "model": self.name,
            "generation_time_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _block_repeated_ngrams(scores: torch.Tensor, seqs: list[list[int]], n: int) -> None:
        """Set -inf for every next token that would repeat an n-gram already in the beam."""
        for i, seq in enumerate(seqs):
            prefix = tuple(seq[len(seq) - n + 1:])  # last n-1 tokens
            banned = {
                seq[j + n - 1]
                for j in range(1, len(seq) - n + 1)  # skip <start> at index 0
                if tuple(seq[j:j + n - 1]) == prefix
            }
            if banned:
                scores[i, list(banned)] = float("-inf")

    def _beam_search(self, image: torch.Tensor, beam_size: int, max_len: int):
        """Keep the best partial captions at every decoding step."""
        k = beam_size
        vocab_size = len(self.word_map)
        decoder = self.decoder
        start_idx, end_idx = self.word_map["<start>"], self.word_map["<end>"]
        unk_idx = self.word_map.get("<unk>")
        n = self.no_repeat_ngram

        encoder_out = self.encoder(image)  # (1, 14, 14, 2048)
        enc_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        k_prev_words = torch.full((k, 1), start_idx, dtype=torch.long, device=self.device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1, device=self.device)
        seqs_alpha = torch.ones(k, 1, enc_size, enc_size, device=self.device)

        complete_seqs, complete_alphas, complete_scores = [], [], []
        h, c = decoder.init_hidden_state(encoder_out)
        step = 1

        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)
            awe, alpha = decoder.attention(encoder_out, h)
            alpha = alpha.view(-1, enc_size, enc_size)
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * awe
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))

            scores = F.log_softmax(decoder.fc(h), dim=1)
            if unk_idx is not None:
                scores[:, unk_idx] = float("-inf")  # never emit <unk>
            if n > 0 and seqs.size(1) >= n:
                self._block_repeated_ngrams(scores, seqs.tolist(), n)
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_inds = torch.div(top_k_words, vocab_size, rounding_mode="floor")
            next_inds = top_k_words % vocab_size

            seqs = torch.cat([seqs[prev_inds], next_inds.unsqueeze(1)], dim=1)
            seqs_alpha = torch.cat([seqs_alpha[prev_inds], alpha[prev_inds].unsqueeze(1)], dim=1)

            incomplete = [i for i, w in enumerate(next_inds.tolist()) if w != end_idx]
            complete = [i for i in range(len(next_inds)) if i not in incomplete]

            if complete:
                complete_seqs.extend(seqs[complete].tolist())
                complete_alphas.extend(seqs_alpha[complete].tolist())
                complete_scores.extend(top_k_scores[complete].tolist())
            k -= len(complete)

            if k == 0 or step >= max_len:
                break

            seqs = seqs[incomplete]
            seqs_alpha = seqs_alpha[incomplete]
            h = h[prev_inds[incomplete]]
            c = c[prev_inds[incomplete]]
            encoder_out = encoder_out[prev_inds[incomplete]]
            top_k_scores = top_k_scores[incomplete].unsqueeze(1)
            k_prev_words = next_inds[incomplete].unsqueeze(1)
            step += 1

        if not complete_seqs:  # length limit reached before any beam produced <end>
            best = int(top_k_scores.view(-1).argmax())
            return seqs[best].tolist(), seqs_alpha[best].tolist()

        best = complete_scores.index(max(complete_scores))
        return complete_seqs[best], complete_alphas[best]
