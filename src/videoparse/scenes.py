"""Stage 6: group shots into thematic scenes (multimodal TextTiling over shot boundaries).

At every shot boundary we compare what comes before with what comes after on three signals:
  visual  - best SigLIP2 match between the w shots on each side    (same set / camera setups)
  text    - bge embedding of the transcript 60 s on each side       (same topic)
  people  - screen time per face group 60 s on each side            (same participants)
Each signal is z-scored, combined, and turned into a TextTiling depth score. Cuts are the
deepest boundaries, subject to minimum/maximum scene durations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Ctx, log, write_json
from .embed import embed_texts


def run(ctx: Ctx) -> None:
    cfg = ctx.cfg["scenes"]
    meta, shots = ctx.meta(), ctx.shots()
    duration = meta["duration"]
    bounds = boundary_scores(ctx)
    write_json(ctx.cache / "boundaries.json", bounds)

    cut_times = select_cuts(bounds, duration, cfg, chapters=[c["start"] for c in meta.get("chapters", [])])
    scenes = scenes_from_cuts(shots, snap_establishing(ctx, cut_times))
    write_json(ctx.cache / "scenes_local.json", scenes)
    write_json(ctx.cache / "scenes.json", scenes)
    write_json(ctx.cache / "refine_candidates.json", candidates(bounds, cut_times, cfg))
    lens = [s["end_seconds"] - s["start_seconds"] for s in scenes]
    log.info("scenes (local): %d scenes, length min %.0fs / median %.0fs / max %.0fs", len(scenes),
             min(lens), float(np.median(lens)), max(lens))


def boundary_scores(ctx: Ctx) -> list[dict]:
    cfg = ctx.cfg["scenes"]
    shots = ctx.shots()
    V = np.load(ctx.cache / "shot_visual.npy")
    S = V @ V.T
    segs = ctx_transcript(ctx)
    faces = _faces(ctx)
    w, tw = cfg["visual_window_shots"], cfg["text_window_seconds"]
    n = len(shots)

    rows, left_txt, right_txt = [], [], []
    for b in range(1, n):
        tau = shots[b]["start"]
        # Best-matching shot pair across the boundary: inside a scene the camera returns to the same
        # setups (shot / reverse shot), across a scene change no shot on the right resembles the left.
        sim_v = float(S[max(0, b - w):b, b:min(n, b + w)].max())
        left_txt.append(_text_in(segs, tau - tw, tau))
        right_txt.append(_text_in(segs, tau, tau + tw))
        sim_p = _cos(_people(faces, tau - tw, tau), _people(faces, tau, tau + tw))
        rows.append({"shot_id": b, "time": tau, "sim_v": sim_v, "sim_p": sim_p})

    if rows:
        texts = sorted({t for t in left_txt + right_txt if t})
        E = dict(zip(texts, embed_texts(ctx, texts))) if texts else {}
        for r, lt, rt in zip(rows, left_txt, right_txt):
            r["sim_t"] = float(E[lt] @ E[rt]) if lt and rt else float("nan")

    weights = cfg["weights"]
    z = {k: _z(np.array([r[f"sim_{k[0]}"] for r in rows], float)) for k in ("visual", "text", "people")}
    for i, r in enumerate(rows):
        num = den = 0.0
        for k, wk in weights.items():
            if not np.isnan(z[k][i]):
                num += wk * z[k][i]
                den += wk
        r["sim"] = num / den if den else 0.0
    sim = np.array([r["sim"] for r in rows])
    for i, r in enumerate(rows):
        r["depth"] = _depth(sim, i)
    return rows


def select_cuts(bounds: list[dict], duration: float, cfg: dict, chapters: list[float] = ()) -> list[float]:
    """Deepest boundaries first, keeping every scene >= min length; then split over-long scenes."""
    if not bounds:
        return []
    min_len, max_len = cfg["min_scene_seconds"], cfg["max_scene_seconds"]
    depth = np.array([b["depth"] for b in bounds])
    times = np.array([b["time"] for b in bounds])
    floor = depth.mean() - 0.5 * depth.std()
    n_cuts = max(0, round(duration / cfg["target_scene_seconds"]) - 1)

    # YouTube chapters are only a hint (auto-generated chapters are often a few seconds off a real
    # scene change): boundaries near a chapter start get a bonus of chapter_bonus standard deviations.
    for c in chapters:
        if c and c > 0:
            near = np.abs(times - c) <= cfg["chapter_snap_seconds"]
            depth = depth + near * cfg["chapter_bonus"] * depth.std()
    forced: set = set()
    order = list(np.argsort(-depth))

    def fits(t, cuts):
        prev = max([c for c in cuts if c < t], default=0.0)
        nxt = min([c for c in cuts if c > t], default=duration)
        return t - prev >= min_len and nxt - t >= min_len

    cuts: list[float] = []
    for i in order:
        if i not in forced and (len(cuts) >= n_cuts or depth[i] < floor):
            break
        if fits(times[i], cuts):
            cuts.append(float(times[i]))

    # Split scenes longer than max_len at their deepest valid internal boundary.
    changed = True
    while changed:
        changed = False
        edges = [0.0, *sorted(cuts), duration]
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a <= max_len:
                continue
            inside = [i for i in range(len(times)) if a + min_len <= times[i] <= b - min_len]
            if inside:
                best = max(inside, key=lambda i: depth[i])
                cuts.append(float(times[best]))
                changed = True
                break
    return sorted(cuts)


def snap_establishing(ctx: Ctx, cut_times: list[float]) -> list[float]:
    """Move each cut back over the faceless shots right before it (establishing exteriors, aerials):
    in TV/film grammar an establishing shot opens the scene it introduces. A faceless shot only moves
    if it looks more like the following shots than the preceding ones, so a closing shot (e.g. the
    wall two characters just walked past) stays with its own scene."""
    cfg, shots = ctx.cfg["scenes"], ctx.shots()
    with_faces = set(pd.read_parquet(ctx.cache / "face_dets.parquet", columns=["shot_id"])["shot_id"])
    V = np.load(ctx.cache / "shot_visual.npy")
    S = V @ V.T
    start_to_id = {round(s["start"], 4): s["id"] for s in shots}
    out, prev = [], 0.0
    for t in sorted(cut_times):
        k = b = start_to_id[round(t, 4)]
        while (k - 1 > 0 and b - (k - 1) <= cfg["establishing_max_shots"] and (k - 1) not in with_faces
               and t - shots[k - 1]["start"] <= cfg["establishing_max_seconds"]
               and shots[k - 1]["start"] - prev >= cfg["min_scene_seconds"]):
            k -= 1
        before, after = S[:, max(0, k - 6):k], S[:, b:b + 6]
        new_k = b
        for j in range(b - 1, k - 1, -1):
            if before.shape[1] and before[j].max() > after[j].max():
                break
            new_k = j
        if new_k != b:
            log.info("  cut %.2fs moved back to %.2fs (%d establishing shot(s))", t, shots[new_k]["start"], b - new_k)
        out.append(shots[new_k]["start"])
        prev = out[-1]
    return out


def scenes_from_cuts(shots: list[dict], cut_times: list[float], titles: list[str] | None = None) -> list[dict]:
    cut_set = {round(t, 4) for t in cut_times}
    scenes, first = [], 0
    for s in shots[1:]:
        if round(s["start"], 4) in cut_set:
            scenes.append((first, s["id"] - 1))
            first = s["id"]
    scenes.append((first, shots[-1]["id"]))
    out = []
    for i, (a, b) in enumerate(scenes):
        sc = {"id": i, "start_seconds": shots[a]["start"], "end_seconds": shots[b]["end"],
              "first_shot": a, "last_shot": b}
        if titles and i < len(titles):
            sc["title"] = titles[i]
        out.append(sc)
    return out


def candidates(bounds: list[dict], cut_times: list[float], cfg: dict) -> list[dict]:
    """Top-depth boundaries spaced >= candidate_spacing_seconds apart, always including the chosen cuts."""
    chosen = {round(t, 4) for t in cut_times}
    picked = [b for b in bounds if round(b["time"], 4) in chosen]
    for b in sorted(bounds, key=lambda b: -b["depth"]):
        if len(picked) >= cfg["refine_candidates"]:
            break
        if all(abs(b["time"] - p["time"]) >= cfg["candidate_spacing_seconds"] for p in picked):
            picked.append(b)
    picked.sort(key=lambda b: b["time"])
    return [{"cid": i + 1, "shot_id": b["shot_id"], "time": b["time"], "depth": round(b["depth"], 3),
             "chosen_locally": round(b["time"], 4) in chosen} for i, b in enumerate(picked)]


# --------------------------------------------------------------------------- helpers

def ctx_transcript(ctx: Ctx) -> list[dict]:
    from .common import read_json

    return read_json(ctx.cache / "transcript.json")["segments"]


def _faces(ctx: Ctx) -> pd.DataFrame:
    p = ctx.cache / "faces.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["ts", "group"])
    df = pd.read_parquet(p, columns=["ts", "group"])
    return df[df["group"] > 0]


def _text_in(segs: list[dict], t0: float, t1: float) -> str:
    return " ".join(s["text"] for s in segs if t0 <= (s["start"] + s["end"]) / 2 < t1)


def _people(faces: pd.DataFrame, t0: float, t1: float) -> np.ndarray:
    groups = int(faces["group"].max()) if len(faces) else 0
    v = np.zeros(max(groups, 1))
    sel = faces[(faces["ts"] >= t0) & (faces["ts"] < t1)]
    for g, cnt in sel["group"].value_counts().items():
        v[int(g) - 1] = cnt
    return v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def _z(x: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(x)
    if ok.sum() < 2 or np.nanstd(x) == 0:
        return np.where(ok, 0.0, np.nan)
    return (x - np.nanmean(x)) / np.nanstd(x)


def _depth(sim: np.ndarray, i: int) -> float:
    """TextTiling depth: climb left/right while similarity keeps rising."""
    l = i
    while l - 1 >= 0 and sim[l - 1] >= sim[l]:
        l -= 1
    r = i
    while r + 1 < len(sim) and sim[r + 1] >= sim[r]:
        r += 1
    return float((sim[l] - sim[i]) + (sim[r] - sim[i]))
