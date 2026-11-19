"""One-time conversion of the pretrained Show-Attend-Tell checkpoint to a clean state_dict.

The original BEST_checkpoint_*.pth.tar pickles whole Encoder/DecoderWithAttention
objects from a module named `models` (plus two Adam optimizers). This script:
  1. aliases `models` -> `model.captioning.sat_models` so the pickle can be resolved,
  2. stubs the optimizers (not needed for inference; their old pickle format
     also fails to load on recent PyTorch),
  3. saves only tensors + word map + metadata to sat_resnet101_coco.pth,
  4. rebuilds fresh models, loads the new file with strict=True and
     weights_only=True, and checks the outputs match the original exactly.

Run from backend/ (with the .pth.tar copied into weights/):
    python -m model.captioning.convert_checkpoint
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend/model/captioning -> backend/
WEIGHTS_DIR = BACKEND_ROOT / "weights"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from model.captioning import sat_models  # noqa: E402
from model.captioning.sat_models import DecoderWithAttention, Encoder  # noqa: E402

SRC = WEIGHTS_DIR / "BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
WORD_MAP = WEIGHTS_DIR / "WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
DST = WEIGHTS_DIR / "sat_resnet101_coco.pth"

DIMS = dict(attention_dim=512, embed_dim=512, decoder_dim=512, encoder_dim=2048)


class _OptimizerStub:
    """Stands in for pickled torch.optim objects; their state is discarded."""

    def __setstate__(self, state):  # noqa: D401
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("torch.optim"):
            return _OptimizerStub
        return super().find_class(module, name)


_pickle_module = types.ModuleType("sat_pickle")
_pickle_module.Unpickler = _Unpickler
_pickle_module.load = pickle.load


def load_original(path: Path, device: torch.device) -> dict:
    """Load the original pickled checkpoint via the models -> sat_models alias."""
    sys.modules["models"] = sat_models
    try:
        return torch.load(path, map_location=device, weights_only=False,
                          pickle_module=_pickle_module)
    finally:
        sys.modules.pop("models", None)


def build_models(vocab_size: int):
    encoder = Encoder(encoded_image_size=14)
    decoder = DecoderWithAttention(vocab_size=vocab_size, **DIMS)
    return encoder.eval(), decoder.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--word-map", type=Path, default=WORD_MAP)
    ap.add_argument("--dst", type=Path, default=DST)
    args = ap.parse_args()

    cpu = torch.device("cpu")
    word_map = json.loads(args.word_map.read_text())
    print(f"word map: {len(word_map)} entries")

    ckpt = load_original(args.src, cpu)
    old_enc, old_dec = ckpt["encoder"].eval(), ckpt["decoder"].eval()
    print(f"loaded original: epoch {ckpt.get('epoch')}, val BLEU-4 {ckpt.get('bleu-4')}")

    if old_dec.embedding.num_embeddings != len(word_map):
        raise SystemExit(
            f"vocab mismatch: decoder {old_dec.embedding.num_embeddings} vs word map {len(word_map)}"
        )

    payload = {
        "encoder_state": {k: v.clone() for k, v in old_enc.state_dict().items()},
        "decoder_state": {k: v.clone() for k, v in old_dec.state_dict().items()},
        "word_map": word_map,
        "meta": {
            "source": "pretrained Show-Attend-Tell checkpoint",
            "source_file": args.src.name,
            "epoch": int(ckpt.get("epoch", -1)),
            "val_bleu4": float(ckpt.get("bleu-4", float("nan"))),
            "encoded_image_size": 14,
            "vocab_size": len(word_map),
            **DIMS,
        },
    }
    torch.save(payload, args.dst)
    print(f"saved {args.dst} ({args.dst.stat().st_size / 1e6:.0f} MB)")

    # --- verify: fresh models + safe load + identical outputs -----------------
    clean = torch.load(args.dst, map_location=cpu, weights_only=True)
    enc, dec = build_models(len(clean["word_map"]))
    enc.load_state_dict(clean["encoder_state"], strict=True)
    dec.load_state_dict(clean["decoder_state"], strict=True)

    torch.manual_seed(0)
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        a, b = old_enc(x), enc(x)
        flat_a, flat_b = a.view(1, -1, 2048), b.view(1, -1, 2048)
        ha, _ = old_dec.init_hidden_state(flat_a)
        hb, _ = dec.init_hidden_state(flat_b)
        _, alpha_a = old_dec.attention(flat_a, ha)
        _, alpha_b = dec.attention(flat_b, hb)
        la, lb = old_dec.fc(ha), dec.fc(hb)
    diff = max((a - b).abs().max().item(), (alpha_a - alpha_b).abs().max().item(),
               (la - lb).abs().max().item())
    print(f"encoder output {tuple(b.shape)}; max abs diff vs original: {diff:.2e}")
    if diff > 1e-5:
        raise SystemExit("verification FAILED: converted model differs from original")
    print("verification OK")


if __name__ == "__main__":
    main()
