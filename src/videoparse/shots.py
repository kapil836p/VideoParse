"""Stage 2: shot boundary detection (TransNetV2 on CPU, PySceneDetect fallback)."""

from __future__ import annotations

import cv2
import numpy as np

from .common import Ctx, log, write_json


def run(ctx: Ctx) -> None:
    meta = ctx.meta()
    fps, duration = meta["fps"], meta["duration"]
    cfg = ctx.cfg["shots"]
    try:
        bounds, thumbs = _transnet(ctx, cfg["threshold"], fps)
        method = "transnetv2"
    except Exception as e:  # noqa: BLE001 - any failure -> fallback detector
        log.warning("TransNetV2 failed (%s); falling back to PySceneDetect", e)
        bounds, thumbs, method = _pyscenedetect(ctx), None, "pyscenedetect"

    cuts = sorted({round(b, 4) for b in bounds if 0 < b < duration})
    edges = [0.0, *cuts, duration]
    shots = [{"start": a, "end": b} for a, b in zip(edges[:-1], edges[1:]) if b > a]
    if thumbs is not None:
        shots = _merge_tiny(shots, thumbs, fps, cfg["min_shot_seconds"], cfg["merge_hist_dist"])
    for i, s in enumerate(shots):
        s["id"] = i
    write_json(ctx.cache / "shots.json", shots)
    lens = np.array([s["end"] - s["start"] for s in shots])
    log.info("%s: %d shots (median %.1fs, min %.2fs, max %.1fs)", method, len(shots),
             np.median(lens), lens.min(), lens.max())


def _transnet(ctx: Ctx, threshold: float, fps: float) -> tuple[list[float], np.ndarray]:
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device="cpu")  # MPS gives run-to-run inconsistent results
    model.eval()
    import torch

    with torch.no_grad():
        frames, single, _ = model.predict_video(str(ctx.video), quiet=True)
    single = _np(single).reshape(-1)
    frames = _np(frames)
    np.save(ctx.cache / "transnet_pred.npy", single)

    # Each run of frames above threshold is one transition; the boundary sits at its midpoint.
    # A hard cut at frame i (single-frame run) means the new shot starts at frame i + 1.
    on = single > threshold
    bounds, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            bounds.append((round((i + j) / 2) + 1) / fps)
            i = j + 1
        else:
            i += 1
    return bounds, frames


def _pyscenedetect(ctx: Ctx) -> list[float]:
    from scenedetect import AdaptiveDetector, detect

    scenes = detect(str(ctx.video), AdaptiveDetector(adaptive_threshold=3.0, min_content_val=15))
    return [s[0].get_seconds() for s in scenes[1:]]


def _merge_tiny(shots, thumbs, fps, min_len, max_dist):
    """Merge very short shots into the neighbour they look like; keep genuine fast cuts."""

    def hist(s):
        a = int(s["start"] * fps)
        b = max(a + 1, int(s["end"] * fps))
        mid = thumbs[min((a + b) // 2, len(thumbs) - 1)]
        hsv = cv2.cvtColor(np.ascontiguousarray(mid), cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten()

    changed = True
    while changed:
        changed = False
        for i, s in enumerate(shots):
            if s["end"] - s["start"] >= min_len or len(shots) == 1:
                continue
            hs = hist(s)
            cands = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(shots):
                    d = cv2.compareHist(hs, hist(shots[j]), cv2.HISTCMP_BHATTACHARYYA)
                    cands.append((d, j))
            d, j = min(cands)
            if d < max_dist:
                a, b = sorted((i, j))
                shots[a] = {"start": shots[a]["start"], "end": shots[b]["end"]}
                del shots[b]
                changed = True
                break
    return shots


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)
