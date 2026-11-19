"""
CATRCaptioner: CAption TRansformer.

ResNet-101 (dilated, stride 16) + 6-layer transformer encoder/decoder,
BERT word-piece vocabulary, trained on COCO 2017.
Model code: model/captioning/catr/. Weights: weights/catr_v3_coco.pth.

Per request:
  1. Resize so the longest side is 299 px (keeps aspect ratio), normalise to [-1, 1].
  2. Backbone + transformer encoder run ONCE -> memory over an H x W grid.
  3. Beam search over the decoder (the reference greedy path re-runs the
     whole network every step; this gives the same greedy result much faster).
  4. One extra decoder pass on the final caption with hooks on every decoder
     layer's cross-attention -> one H x W map per word (mean over heads/layers).
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
from PIL import Image

from ...config import BACKEND_ROOT
from .common import block_repeated_ngrams, merge_wordpieces, normalise_map, select_device

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from model.captioning.catr import caption as catr_caption  # noqa: E402
from model.captioning.catr.configuration import Config as CATRConfig  # noqa: E402
from model.captioning.catr.transformer import generate_square_subsequent_mask  # noqa: E402
from model.captioning.catr.utils import nested_tensor_from_tensor_list  # noqa: E402

log = logging.getLogger("backend.caption.catr")


class CATRCaptioner:
    name = "catr"
    label = "CATR (ResNet-101 + Transformer, COCO 2017)"

    def __init__(self, cfg: dict) -> None:
        own = cfg.get("catr") or {}
        weights = Path(own.get("weights", "weights/catr_v3_coco.pth"))
        tok_dir = Path(own.get("tokenizer", "weights/catr_tokenizer"))
        self.checkpoint_path = weights if weights.is_absolute() else BACKEND_ROOT / weights
        self.tokenizer_dir = tok_dir if tok_dir.is_absolute() else BACKEND_ROOT / tok_dir
        self.max_side = int(own.get("max_image_side", 299))
        self.default_beam_size = int(own.get("beam_size", cfg.get("beam_size", 3)))
        self.max_caption_len = int(own.get("max_caption_len", cfg.get("max_caption_len", 50)))
        self.no_repeat_ngram = int(own.get("no_repeat_ngram", cfg.get("no_repeat_ngram", 2)))
        self.length_alpha = float(own.get("length_alpha", 0.7))

        self.device = select_device()
        self.lock = threading.Lock()
        self.ready = False
        self.error: str | None = None
        self.vocab_size = 0
        self.model = None
        self.tokenizer = None
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            self.error = (f"CATR weights not found: {self.checkpoint_path}. Download "
                          "https://github.com/saahiluppal/catr/releases/download/0.2/weight493084032.pth")
            log.warning(self.error)
            return
        try:
            from transformers import BertTokenizer

            model, _ = catr_caption.build_model(CATRConfig())
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
            if self.tokenizer_dir.exists():
                tokenizer = BertTokenizer.from_pretrained(str(self.tokenizer_dir))
            else:  # first run: fetch bert-base-uncased vocab (~230 KB) and keep a local copy
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                tokenizer.save_pretrained(str(self.tokenizer_dir))
            self.model = model.to(self.device).eval()
            self.tokenizer = tokenizer
            self.vocab_size = tokenizer.vocab_size
            self.ready = True
            self.caption(Image.new("RGB", (self.max_side, self.max_side), (128, 128, 128)),
                         beam_size=self.default_beam_size, max_len=8)  # warm-up
            log.info("CATR loaded on %s", self.device)
        except Exception as exc:
            self.ready = False
            self.error = f"Failed to load CATR: {exc}"
            log.exception(self.error)

    # ------------------------------------------------------------------
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        img = image.convert("RGB")
        w, h = img.size
        scale = self.max_side / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC,
                         reducing_gap=3.0)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(arr).permute(2, 0, 1).to(self.device)

    def _encode(self, x: torch.Tensor):
        model = self.model
        samples = nested_tensor_from_tensor_list([x])
        feats, pos = model.backbone(samples)
        src, mask = feats[-1].decompose()
        src = model.input_proj(src)
        h, w = src.shape[-2:]
        src = src.flatten(2).permute(2, 0, 1)
        pos = pos[-1].flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = model.transformer.encoder(src, src_key_padding_mask=mask, pos=pos)
        return memory, mask, pos, (int(h), int(w))

    def _decode(self, memory, mask, pos, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (n, t) -> decoder output (t, n, C)."""
        T = self.model.transformer
        n, t = tokens.shape
        tgt = T.embeddings(tokens).permute(1, 0, 2)
        query = T.embeddings.position_embeddings.weight[:t].unsqueeze(1).repeat(1, n, 1)
        return T.decoder(tgt, memory.expand(-1, n, -1), memory_key_padding_mask=mask.expand(n, -1),
                         pos=pos.expand(-1, n, -1), query_pos=query,
                         tgt_mask=generate_square_subsequent_mask(t).to(tokens.device))

    def _beam_search(self, memory, mask, pos, beam: int, max_len: int):
        tok = self.tokenizer
        cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
        banned = [tok.pad_token_id, tok.unk_token_id, cls_id, tok.mask_token_id]
        seqs = [[cls_id]]
        scores = torch.zeros(1, device=self.device)
        done: list[tuple[float, list[int]]] = []
        for _ in range(max_len):
            tokens = torch.tensor(seqs, device=self.device)
            hs = self._decode(memory, mask, pos, tokens)
            logp = F.log_softmax(self.model.mlp(hs[-1]), dim=-1)  # (n, V)
            logp[:, banned] = float("-inf")
            block_repeated_ngrams(logp, seqs, self.no_repeat_ngram, skip=1)
            total = (scores[:, None] + logp).view(-1)
            top, idx = total.topk(min(beam, total.numel()))
            V = logp.shape[1]
            new_seqs, new_scores = [], []
            for score, flat in zip(top.tolist(), idx.tolist()):
                parent, word = divmod(flat, V)
                seq = seqs[parent] + [word]
                if word == sep_id:
                    length = len(seq) - 2  # without CLS / SEP
                    done.append((score / max(length, 1) ** self.length_alpha, seq[1:-1]))
                else:
                    new_seqs.append(seq)
                    new_scores.append(score)
            if not new_seqs or len(done) >= beam:
                break
            seqs, scores = new_seqs, torch.tensor(new_scores, device=self.device)
        if not done:  # length limit reached: take the best unfinished beam
            done = [(float(s) / max(len(q) - 1, 1) ** self.length_alpha, q[1:])
                    for s, q in zip(scores.tolist(), seqs)]
        return max(done, key=lambda d: d[0])[1]

    def _attention(self, memory, mask, pos, ids: list[int], hw) -> list[np.ndarray]:
        """Cross-attention of each generated token over the H x W grid (mean of heads and layers)."""
        captured = []
        hooks = [layer.multihead_attn.register_forward_hook(lambda m, i, o: captured.append(o[1]))
                 for layer in self.model.transformer.decoder.layers]
        try:
            tokens = torch.tensor([[self.tokenizer.cls_token_id] + ids], device=self.device)
            self._decode(memory, mask, pos, tokens)
        finally:
            for hook in hooks:
                hook.remove()
        att = torch.stack([a[0] for a in captured if a is not None]).mean(0)  # (t, S)
        h, w = hw
        # position i (input token i) predicts token i+1 -> rows 0..len(ids)-1
        return [att[i].reshape(h, w).float().cpu().numpy() for i in range(len(ids))]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def caption(self, image: Image.Image, beam_size: int | None = None,
                max_len: int | None = None) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "CATR is not loaded.")
        started = time.perf_counter()
        beam = int(beam_size or self.default_beam_size)
        limit = int(max_len or self.max_caption_len)
        with self.lock:
            memory, mask, pos, hw = self._encode(self._preprocess(image))
            ids = self._beam_search(memory, mask, pos, beam, limit)
            maps = self._attention(memory, mask, pos, ids, hw) if ids else []
        pieces = self.tokenizer.convert_ids_to_tokens(ids)
        words, word_maps = merge_wordpieces(pieces, maps)
        keep = [i for i, wd in enumerate(words) if any(ch.isalnum() for ch in wd)]
        words = [words[i] for i in keep]
        alphas = [normalise_map(word_maps[i]) for i in keep]
        return {
            "caption": " ".join(words),
            "words": words,
            "alphas": alphas,
            "grid_size": hw[0],
            "grid_shape": [hw[0], hw[1]],
            "generation_time_ms": round((time.perf_counter() - started) * 1000, 1),
            "model": self.name,
        }
