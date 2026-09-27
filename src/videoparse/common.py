"""Shared context: paths, config, JSON helpers and frame access."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import yaml

log = logging.getLogger("videoparse")


@dataclass
class Ctx:
    root: Path
    cfg: dict

    @classmethod
    def load(cls, root: Path | str = ".", config: str = "config.yaml") -> "Ctx":
        root = Path(root).resolve()
        cfg = yaml.safe_load((root / config).read_text())
        return cls(root=root, cfg=cfg)

    @property
    def data(self) -> Path:
        return self._dir("data")

    @property
    def cache(self) -> Path:
        return self._dir("cache")

    @property
    def output(self) -> Path:
        return self._dir("output")

    @property
    def video(self) -> Path:
        return self.data / "video.mp4"

    @property
    def audio(self) -> Path:
        return self.data / "audio.wav"

    def _dir(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def meta(self) -> dict:
        return read_json(self.cache / "meta.json")

    def shots(self) -> list[dict]:
        return read_json(self.cache / "shots.json")

    def scenes(self) -> list[dict]:
        return read_json(self.cache / "scenes.json")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, obj, indent: int | None = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=indent, ensure_ascii=False) + "\n")


def hms(seconds: float) -> str:
    """452.5 -> '00:07:32.50'"""
    h, rem = divmod(max(seconds, 0.0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def shot_index(shots: list[dict], t: float) -> int:
    """Index of the shot containing time t (shots tile the video)."""
    starts = np.array([s["start"] for s in shots])
    return int(np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(shots) - 1))


def iter_frames_at(video: Path, times: Iterable[float], fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t, BGR frame) for each requested time, decoding the video once, sequentially.

    Frames that are not needed are only grabbed (not converted), which is much
    faster than random seeking on H.264.
    """
    wanted: dict[int, list[float]] = {}
    for t in sorted(times):
        wanted.setdefault(int(round(t * fps)), []).append(t)
    if not wanted:
        return
    last = max(wanted)
    cap = cv2.VideoCapture(str(video))
    idx = 0
    try:
        while idx <= last:
            if not cap.grab():
                break
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    for t in wanted[idx]:
                        yield t, frame
            idx += 1
    finally:
        cap.release()


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
