"""Stage 3: three keyframes per shot (25/50/75%), one for shots under 1 s."""

from __future__ import annotations

import cv2

from .common import Ctx, iter_frames_at, log, resize_max, write_json


def run(ctx: Ctx) -> None:
    meta, shots = ctx.meta(), ctx.shots()
    out = ctx.cache / "keyframes"
    out.mkdir(exist_ok=True)
    last_t = meta["duration"] - 1.5 / meta["fps"]

    wanted: dict[float, tuple[int, int]] = {}
    for s in shots:
        length = s["end"] - s["start"]
        fracs = (0.5,) if length < 1.0 else (0.25, 0.5, 0.75)
        for k, f in enumerate(fracs):
            wanted[min(s["start"] + f * length, last_t)] = (s["id"], k)

    index: dict[int, list[str]] = {s["id"]: [] for s in shots}
    for t, frame in iter_frames_at(ctx.video, wanted, meta["fps"]):
        sid, k = wanted[t]
        path = out / f"shot_{sid:04d}_{k}.jpg"
        cv2.imwrite(str(path), resize_max(frame, ctx.cfg["keyframes"]["max_side"]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        index[sid].append(str(path.relative_to(ctx.root)))
    write_json(ctx.cache / "keyframes.json", {str(k): sorted(v) for k, v in index.items()})
    missing = [k for k, v in index.items() if not v]
    log.info("keyframes: %d images for %d shots (%d shots without frames)",
             sum(map(len, index.values())), len(shots), len(missing))
