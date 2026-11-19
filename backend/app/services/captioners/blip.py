"""
BLIPCaptioner: BLIP image captioning, base model.

ViT-B/16 image encoder + BERT-style text decoder with cross-attention,
fine-tuned for captioning on COCO.

Per request:
  1. BlipProcessor: resize to 384 x 384, CLIP-style normalisation.
  2. model.generate with beam search and repeat blocking.
  3. One extra text-decoder pass on the final caption with output_attentions=True:
     cross-attention of each token over the 24 x 24 image patches
     (mean over heads and layers; the [CLS] image token is dropped).
"""
from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np
import torch
from PIL import Image

from .common import merge_wordpieces, normalise_map, select_device

log = logging.getLogger("backend.caption.blip")


class BLIPCaptioner:
    name = "blip"
    label = "BLIP-base (ViT-B/16 + text decoder, COCO fine-tuned)"

    def __init__(self, cfg: dict) -> None:
        own = cfg.get("blip") or {}
        self.model_id = own.get("model_id", "Salesforce/blip-image-captioning-base")
        self.checkpoint_path = f"huggingface:{self.model_id}"
        self.default_beam_size = int(own.get("beam_size", cfg.get("beam_size", 3)))
        self.max_caption_len = int(own.get("max_caption_len", min(int(cfg.get("max_caption_len", 30)), 40)))
        self.no_repeat_ngram = int(own.get("no_repeat_ngram", cfg.get("no_repeat_ngram", 2)))
        self.repetition_penalty = float(own.get("repetition_penalty", 1.0))

        self.device = select_device()
        self.lock = threading.Lock()
        self.ready = False
        self.error: str | None = None
        self.vocab_size = 0
        self.model = None
        self.processor = None
        self._load()

    def _load(self) -> None:
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            try:  # use the local cache without a network round-trip when possible
                processor = BlipProcessor.from_pretrained(self.model_id, local_files_only=True)
                model = BlipForConditionalGeneration.from_pretrained(self.model_id, local_files_only=True)
            except OSError:
                log.info("Downloading %s (~1 GB, first run only)", self.model_id)
                processor = BlipProcessor.from_pretrained(self.model_id)
                model = BlipForConditionalGeneration.from_pretrained(self.model_id)
            self.processor = processor
            self.model = model.to(self.device).eval()
            self.vocab_size = processor.tokenizer.vocab_size
            self.ready = True
            self.caption(Image.new("RGB", (384, 384), (128, 128, 128)), max_len=8)  # warm-up
            log.info("BLIP loaded on %s", self.device)
        except Exception as exc:
            self.ready = False
            self.error = f"Failed to load BLIP: {exc}"
            log.exception(self.error)

    @torch.no_grad()
    def caption(self, image: Image.Image, beam_size: int | None = None,
                max_len: int | None = None) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "BLIP is not loaded.")
        started = time.perf_counter()
        beam = int(beam_size or self.default_beam_size)
        limit = int(max_len or self.max_caption_len)
        tok = self.processor.tokenizer
        with self.lock:
            inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, num_beams=beam, max_new_tokens=limit,
                                      no_repeat_ngram_size=self.no_repeat_ngram or 0,
                                      repetition_penalty=self.repetition_penalty)
            seq = out[0]
            special = set(tok.all_special_ids)
            # Skip special tokens, including BLIP's decoder-start token, which
            # is outside the tokenizer vocabulary (it maps to None).
            pieces_all = tok.convert_ids_to_tokens(seq.tolist())
            keep_pos = [i for i, (t, p) in enumerate(zip(seq.tolist(), pieces_all))
                        if i > 0 and t not in special and p is not None]
            ids = [int(seq[i]) for i in keep_pos]
            maps: list[np.ndarray] = []
            side = 0
            if ids:
                image_embeds = self.model.vision_model(pixel_values=inputs["pixel_values"])[0]
                dec = self.model.text_decoder(
                    input_ids=seq.unsqueeze(0),
                    attention_mask=torch.ones_like(seq).unsqueeze(0),
                    encoder_hidden_states=image_embeds,
                    output_attentions=True,
                    return_dict=True,
                )
                cross = torch.stack(list(dec.cross_attentions))  # (layers, 1, heads, T, 1+P)
                att = cross.mean(dim=(0, 2))[0][:, 1:]           # (T, P), drop the [CLS] image token
                # ViT "attention sinks": a few patches take most of the attention for
                # EVERY token, so raw maps all peak at the same spot. Keep only what is
                # specific to each word: its attention above the caption's average.
                att = (att - att[keep_pos].mean(0, keepdim=True) if keep_pos else att).clamp(min=0)
                side = int(math.isqrt(att.shape[-1]))
                # the token at position i was predicted from position i-1
                maps = [att[i - 1].reshape(side, side).float().cpu().numpy() for i in keep_pos]
        words, word_maps = merge_wordpieces(tok.convert_ids_to_tokens(ids), maps)
        keep = [i for i, wd in enumerate(words) if any(ch.isalnum() for ch in wd)]
        words = [words[i] for i in keep]
        return {
            "caption": " ".join(words),
            "words": words,
            "alphas": [normalise_map(word_maps[i]) for i in keep],
            "grid_size": side,
            "grid_shape": [side, side],
            "generation_time_ms": round((time.perf_counter() - started) * 1000, 1),
            "model": self.name,
        }
