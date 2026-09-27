"""Per-shot visual embeddings (SigLIP2) and a text-embedding helper (bge-small)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from .common import Ctx, log, read_json


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run(ctx: Ctx) -> None:
    import open_clip

    cfg = ctx.cfg["embed"]
    model, _, preprocess = open_clip.create_model_and_transforms(cfg["siglip_model"], pretrained=cfg["siglip_pretrained"])
    model = model.to(_device()).eval()
    index = read_json(ctx.cache / "keyframes.json")
    shots = ctx.shots()
    feats: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for s in shots:
            paths = index.get(str(s["id"]), [])
            if not paths:
                continue
            batch = torch.stack([preprocess(Image.open(ctx.root / p).convert("RGB")) for p in paths]).to(_device())
            f = torch.nn.functional.normalize(model.encode_image(batch).float(), dim=-1).mean(0)
            feats[s["id"]] = torch.nn.functional.normalize(f, dim=0).cpu().numpy()
    dim = len(next(iter(feats.values())))
    V = np.zeros((len(shots), dim), np.float32)
    for sid, f in feats.items():
        V[sid] = f
    np.save(ctx.cache / "shot_visual.npy", V)
    log.info("visual embeddings: %s", V.shape)


@lru_cache(maxsize=1)
def _text_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=_device())


def embed_texts(ctx: Ctx, texts: list[str]) -> np.ndarray:
    model = _text_model(ctx.cfg["embed"]["text_model"])
    return model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
