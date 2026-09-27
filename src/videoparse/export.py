"""Stage 9: write output/output.json (exact assignment schema), output_extended.json, faces tree."""

from __future__ import annotations

import os
import shutil
import subprocess

import pandas as pd

from .common import Ctx, hms, log, read_json, write_json


def run(ctx: Ctx) -> None:
    meta, shots, scenes = ctx.meta(), ctx.shots(), ctx.scenes()
    desc_path = ctx.cache / "descriptions.json"
    desc = read_json(desc_path) if desc_path.exists() else {}
    interval = ctx.cfg["faces"]["sample_interval"]

    df = pd.read_parquet(ctx.cache / "faces.parquet")
    df = df[df["group"] > 0]
    dup = df.duplicated(["group", "ts"], keep=False)
    if dup.any():  # should not happen (cannot-link), but never let it break the 1:1 file mapping
        log.warning("%d detections share (group, ts); keeping the highest score", int(dup.sum()))
        df = df.sort_values("det_score", ascending=False).drop_duplicates(["group", "ts"])

    faces_dir = ctx.output / "faces"
    shutil.rmtree(faces_dir, ignore_errors=True)
    faces_dir.mkdir(parents=True)

    out_scenes, ext_scenes = [], []
    for sc in scenes:
        a, b = sc["start_seconds"], sc["end_seconds"]
        sel = df[(df["ts"] >= a) & (df["ts"] < b)]
        sdir = faces_dir / f"scene_{sc['id']:03d}"
        sdir.mkdir()
        faces_map, faces_hms, intervals = {}, {}, {}
        for g in sorted(sel["group"].unique()):
            gd = sel[sel["group"] == g].sort_values("ts")
            gdir = sdir / f"group{g}"
            gdir.mkdir()
            for det_id, ts in zip(gd["det_id"], gd["ts"]):
                shutil.copy2(ctx.cache / "crops" / f"{det_id:06d}.jpg", gdir / f"t{ts:07.2f}.jpg")
            ts_list = [round(float(t), 2) for t in gd["ts"]]
            key = f"group{g}"
            faces_map[key] = ts_list
            faces_hms[key] = [hms(t) for t in ts_list]
            intervals[key] = _intervals(ts_list, 1.5 * interval)
        if not faces_map:
            (sdir / "_no_faces").write_text("No faces were detected in this scene.\n")

        d = desc.get(str(sc["id"]), {})
        out_scenes.append({"id": sc["id"], "start_seconds": round(a, 3), "end_seconds": round(b, 3),
                           "description": d.get("description", ""), "faces": faces_map})
        ext_scenes.append({**out_scenes[-1], "start_hms": hms(a), "end_hms": hms(b),
                           "shot_ids": [sc["first_shot"], sc["last_shot"]],
                           "title": d.get("title") or sc.get("title"),
                           "central_theme": d.get("central_theme"), "key_contributors": d.get("key_contributors"),
                           "events": d.get("events"), "description_model": d.get("model"),
                           "faces_hms": faces_hms, "face_intervals": intervals})

    out_shots = [{"id": s["id"], "start_seconds": round(s["start"], 3), "end_seconds": round(s["end"], 3)}
                 for s in shots]
    write_json(ctx.output / "output.json", {"shots": out_shots, "scenes": out_scenes})

    groups = read_json(ctx.cache / "face_groups.json").get("groups", {})
    write_json(ctx.output / "output_extended.json", {
        "video": {k: meta.get(k) for k in ("url", "title", "channel", "duration", "fps", "width", "height")},
        "face_groups": groups,
        "scene_method": "gemini-refined" if (ctx.cache / "refine.json").exists() and _refined(ctx) else "local",
        "shots": out_shots,
        "scenes": ext_scenes,
    })
    _tree(ctx, faces_dir)
    n_files = sum(len(f) for _, _, f in os.walk(faces_dir))
    log.info("export: %d shots, %d scenes, %d face files -> %s", len(out_shots), len(out_scenes), n_files,
             ctx.output / "output.json")


def _refined(ctx: Ctx) -> bool:
    return read_json(ctx.cache / "refine.json").get("cuts") is not None and \
        ctx.scenes() != read_json(ctx.cache / "scenes_local.json")


def _intervals(ts: list[float], gap: float) -> list[list[float]]:
    out: list[list[float]] = []
    for t in ts:
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out


def _tree(ctx: Ctx, faces_dir) -> None:
    tree = subprocess.run(["tree", "-d", "--noreport", "faces"], cwd=ctx.output, capture_output=True, text=True)
    lines = ["# Folder structure (one folder per face group per scene)", tree.stdout.rstrip(), "",
             "# File counts per folder (file name = timecode in seconds, e.g. t0452.00.jpg)"]
    for sdir in sorted(faces_dir.iterdir()):
        groups = sorted((p for p in sdir.iterdir() if p.is_dir()), key=lambda p: int(p.name[5:]))
        counts = ", ".join(f"{g.name}: {len(list(g.glob('*.jpg')))}" for g in groups) or "no faces"
        lines.append(f"{sdir.name}: {counts}")
    (ctx.output / "faces_tree.txt").write_text("\n".join(lines) + "\n")
