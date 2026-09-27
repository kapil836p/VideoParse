"""Stages 7 + 8: Gemini scene-boundary refinement and scene descriptions."""

from __future__ import annotations

import re
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .common import Ctx, hms, log, read_json, write_json
from .faces import best_crops, group_signature
from .gemini import Gemini
from .scenes import scenes_from_cuts, snap_establishing


# --------------------------------------------------------------------------- schemas

class RefineOut(BaseModel):
    keep: list[int] = Field(description="IDs of the candidate cut points (C-numbers) where a new scene starts")


class Contributor(BaseModel):
    group: str | None = Field(description='Face-group label such as "group2", or null for off-screen voices')
    role: str = Field(description="e.g. host, guest, contestant, judge, narrator")
    name: str | None = Field(description="Only if shown on screen or spoken in the video, else null")
    evidence: str | None = Field(description="Where the name comes from, e.g. 'on-screen caption'")


class SceneOut(BaseModel):
    scene_id: int
    title: str = Field(description="At most 8 words")
    central_theme: str = Field(description="One sentence: the main topic of discussion or activity")
    key_contributors: list[Contributor]
    events: list[str] = Field(description="3-6 short sentences, in chronological order")


class DescOut(BaseModel):
    scenes: list[SceneOut]


# --------------------------------------------------------------------------- refinement

def refine(ctx: Ctx) -> None:
    meta, shots = ctx.meta(), ctx.shots()
    cands = read_json(ctx.cache / "refine_candidates.json")
    segs = read_json(ctx.cache / "transcript.json")["segments"]
    faces = _faces(ctx)
    min_len = ctx.cfg["scenes"]["min_scene_seconds"]

    lines = []
    for c in cands:
        t = c["time"]
        before = _groups(faces, t - 60, t)
        after = _groups(faces, t, t + 60)
        lines.append(f"C{c['cid']} at {hms(t)} (score {c['depth']}) | faces before: {before or '-'} | after: "
                     f"{after or '-'}\n   before: \"{_clip(_text(segs, t - 20, t), 250, tail=True)}\"\n"
                     f"   after:  \"{_clip(_text(segs, t, t + 20), 250)}\"")
    chapters = "\n".join(f"- {hms(ch['start'])} {ch['title']}" for ch in meta.get("chapters", [])) or "(none)"
    prompt = (
        f'The video is "{meta["title"]}" by {meta["channel"]}, {hms(meta["duration"])} long.\n'
        "We need to split it into SCENES in the film-editing sense: a scene is a continuous stretch of shots in "
        "one location and time, where one set of people is engaged in one conversation or activity (one theme). "
        "A new scene starts when the location, the time, or the set of people changes, or when the conversation "
        "clearly moves on to a different subject.\n"
        "Below are candidate cut points, each located exactly on a camera cut; a higher score means the visual "
        "setting, transcript topic and people on screen change more there. Watch the video and choose which "
        f"candidates start a new scene. Scenes must be at least {min_len} s long.\n\n"
        "YouTube chapters (coarse and auto-generated; one chapter often contains several scenes, so use them only "
        f"as hints):\n{chapters}\n\nCandidates:\n" + "\n".join(lines)
    )
    gem = Gemini(ctx)
    out, model = gem.ask([("video", (0.0, meta["duration"])), ("text", prompt)], RefineOut)
    by_id = {c["cid"]: c for c in cands}
    kept = sorted(by_id[i]["time"] for i in set(out.keep) if i in by_id)
    kept = _enforce_min(kept, {c["time"]: c["depth"] for c in cands}, meta["duration"], min_len)
    scenes = scenes_from_cuts(shots, snap_establishing(ctx, kept))
    write_json(ctx.cache / "scenes.json", scenes)
    write_json(ctx.cache / "refine.json", {"model": model, "keep": out.keep, "cuts": kept})
    log.info("refine (%s): %d candidates -> %d scenes", model, len(cands), len(scenes))


def _enforce_min(cuts: list[float], depth: dict, duration: float, min_len: float) -> list[float]:
    """Drop the weaker edge of any scene shorter than min_len until all scenes are long enough."""
    cuts = sorted(cuts)
    while cuts:
        edges = [0.0, *cuts, duration]
        short = [(edges[i + 1] - edges[i], i) for i in range(len(edges) - 1) if edges[i + 1] - edges[i] < min_len]
        if not short:
            break
        _, i = min(short)
        removable = [e for e in (edges[i], edges[i + 1]) if e in cuts]
        cuts.remove(min(removable, key=lambda e: depth.get(e, 0.0)))
    return cuts


# --------------------------------------------------------------------------- descriptions

def describe(ctx: Ctx) -> None:
    meta, scenes = ctx.meta(), ctx.scenes()
    segs = read_json(ctx.cache / "transcript.json")["segments"]
    faces = _faces(ctx)
    cfg = ctx.cfg["gemini"]
    interval = ctx.cfg["faces"]["sample_interval"]
    gem = Gemini(ctx)
    sig = group_signature(ctx)  # face regrouping invalidates cached descriptions
    results: dict[int, dict] = {}

    for window in _windows(scenes, cfg["window_seconds"]):
        t0, t1 = window[0]["start_seconds"], window[-1]["end_seconds"]
        groups = _groups(faces, t0, t1, as_list=True)[: cfg["max_gallery_groups"]]
        items: list[tuple] = [("text", _intro(meta, t0, t1, sig))]
        items.append(("video", (t0, t1)))
        for g in groups:
            img = _gallery(ctx, faces, g, t0, t1, cfg["gallery_crops"])
            if img:
                items += [("text", f"Face gallery for group{g}:"), ("image", img)]
        items.append(("text", _scene_block(window, segs, faces, t0, interval, cfg["transcript_chars_per_scene"])))

        want = {s["id"] for s in window}
        got: dict[int, SceneOut] = {}
        try:
            out, model = gem.ask(items, DescOut)
            got = {s.scene_id: s for s in out.scenes if s.scene_id in want}
        except Exception as e:  # noqa: BLE001 - fall through to text-only
            log.warning("video description failed for %s-%s: %s", hms(t0), hms(t1), e)
            model = None
        missing = want - got.keys()
        if missing:
            log.warning("text-only fallback for scenes %s", sorted(missing))
            only = [s for s in window if s["id"] in missing]
            text_items = [("text", _intro(meta, t0, t1, sig, video=False)),
                          ("text", _scene_block(only, segs, faces, t0, interval, cfg["transcript_chars_per_scene"]))]
            out2, model2 = gem.ask(text_items, DescOut, models=gem.models[::-1])
            got |= {s.scene_id: s for s in out2.scenes if s.scene_id in missing}
            model = model or model2
        for s in window:
            if s["id"] not in got:
                raise RuntimeError(f"no description for scene {s['id']}")
            results[s["id"]] = {"out": got[s["id"]], "model": model}
        log.info("described scenes %s", sorted(want))
    gem.cleanup()

    # A person named in one scene keeps that name in every scene (majority vote per face group).
    names = known_names(r["out"] for r in results.values())
    for sid, r in results.items():
        sc = scenes[sid]
        present = {f"group{g}" for g in _groups(faces, sc["start_seconds"], sc["end_seconds"], as_list=True)}
        results[sid] = {**r["out"].model_dump(), "model": r["model"],
                        "description": compose_description(r["out"], present, names)}
    write_json(ctx.cache / "group_names.json", names)
    write_json(ctx.cache / "descriptions.json", {str(k): v for k, v in sorted(results.items())})


def known_names(outs) -> dict[str, str]:
    """Most frequent name given to each face group across all scenes."""
    votes: dict[str, Counter] = {}
    for sd in outs:
        for c in sd.key_contributors:
            g = _norm_group(c.group)
            if g and c.name and c.name.strip():
                votes.setdefault(g, Counter())[c.name.strip()] += 1
    out = {}
    for g, cnt in votes.items():
        # prefer the most frequent name; on ties the longer (fuller) name
        out[g] = max(cnt.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
    return out


def compose_description(sd: SceneOut, present: set[str], names: dict[str, str] | None = None) -> str:
    names = names or {}
    contribs, seen = [], set()
    for c in sd.key_contributors:
        g = _norm_group(c.group)
        role = c.role.strip()
        if g and g in present:
            if g in seen:
                continue
            seen.add(g)
            name = names.get(g) or (c.name or "").strip()
            contribs.append(f"{g} ({name}, {role})" if name else f"{g} ({role})")
        else:
            name = (c.name or "").strip()
            contribs.append(f"{name} ({role}, off-screen)" if name else f"an off-screen {role}")
    events = "; ".join(f"({i}) {e.strip().rstrip('.')}" for i, e in enumerate(sd.events, 1))
    return (f"Central theme: {sd.central_theme.strip().rstrip('.')}. "
            f"Key contributors: {', '.join(contribs) or 'none identified'}. "
            f"Events: {events}.")


# --------------------------------------------------------------------------- helpers

def _intro(meta: dict, t0: float, t1: float, sig: str, video: bool = True) -> str:
    src = (f"The attached video clip is the part of the video from {hms(t0)} to {hms(t1)} (absolute time; "
           "the clip itself starts at 00:00)." if video else
           "The video itself is not available for this request; rely on the transcript and face-group data.")
    return (
        f'You are annotating the YouTube video "{meta["title"]}" by {meta["channel"]}. {src}\n'
        "People visible on screen were detected automatically and clustered by identity into face groups "
        "(group1, group2, ...). A gallery image of each group present follows.\n"
        "For EACH scene listed afterwards, return: title; central_theme; key_contributors (people who speak or drive "
        "the scene; use the face-group label for on-screen people and null for off-screen voices; give a name only "
        "if it is shown on screen or spoken, otherwise null); events (3-6 chronological sentences).\n"
        "Describe only what is seen or heard inside each scene's own time range. "
        f"[groups:{sig}]"
    )


def _scene_block(window, segs, faces, t0, interval, max_chars) -> str:
    out = []
    for s in window:
        a, b = s["start_seconds"], s["end_seconds"]
        gs = faces[(faces["ts"] >= a) & (faces["ts"] < b)]["group"].value_counts()
        people = ", ".join(f"group{g} (~{int(n * interval)} s on screen)" for g, n in gs.items()) or "no faces detected"
        out.append(f"SCENE {s['id']}: {hms(a)}-{hms(b)} absolute = {hms(a - t0)}-{hms(b - t0)} in the clip\n"
                   f"Faces on screen: {people}\nTranscript: \"{_clip(_text(segs, a, b), max_chars)}\"")
    return "\n\n".join(out)


def _windows(scenes: list[dict], max_seconds: float) -> list[list[dict]]:
    windows, cur = [], []
    for s in scenes:
        if cur and s["end_seconds"] - cur[0]["start_seconds"] > max_seconds:
            windows.append(cur)
            cur = []
        cur.append(s)
    if cur:
        windows.append(cur)
    return windows


def _gallery(ctx: Ctx, faces: pd.DataFrame, g: int, t0: float, t1: float, k: int) -> bytes | None:
    full = pd.read_parquet(ctx.cache / "faces.parquet")
    ids = best_crops(full, g, k, t0, t1)
    tiles = [cv2.resize(img, (160, 160)) for i in ids
             if (img := cv2.imread(str(ctx.cache / "crops" / f"{i:06d}.jpg"))) is not None]
    if not tiles:
        return None
    ok, buf = cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else None


def _faces(ctx: Ctx) -> pd.DataFrame:
    df = pd.read_parquet(ctx.cache / "faces.parquet", columns=["ts", "group"])
    return df[df["group"] > 0]


def _groups(faces: pd.DataFrame, t0: float, t1: float, as_list: bool = False):
    gs = faces[(faces["ts"] >= t0) & (faces["ts"] < t1)]["group"].value_counts().index.tolist()
    return [int(g) for g in gs] if as_list else ", ".join(f"group{g}" for g in gs)


def _text(segs: list[dict], t0: float, t1: float) -> str:
    return " ".join(s["text"] for s in segs if t0 <= (s["start"] + s["end"]) / 2 < t1)


def _clip(text: str, n: int, tail: bool = False) -> str:
    if len(text) <= n:
        return text
    return "..." + text[-n:] if tail else text[:n] + "..."


def _norm_group(g: str | None) -> str | None:
    m = re.search(r"group\s*_?(\d+)", g or "", re.I)
    return f"group{int(m.group(1))}" if m else None
