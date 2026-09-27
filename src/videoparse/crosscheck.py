"""Independent cross-verification of the submission (`videoparse verify`).

Uses only output/output.json, the files under output/ and ffprobe; it does not reuse pipeline state,
so it catches bugs in the pipeline's own bookkeeping.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .common import Ctx, log


def run(ctx: Ctx) -> None:
    out_dir = ctx.output
    raw = json.loads((out_dir / "output.json").read_text())
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    # 1. schema: exact keys and types
    check(list(raw) == ["shots", "scenes"], f"top-level keys {list(raw)}")
    for s in raw["shots"]:
        check(list(s) == ["id", "start_seconds", "end_seconds"], f"shot keys {list(s)}")
        check(all(isinstance(s[k], (int, float)) for k in s), f"shot {s['id']} non-numeric")
    for s in raw["scenes"]:
        check(list(s) == ["id", "start_seconds", "end_seconds", "description", "faces"], f"scene keys {list(s)}")
        check(isinstance(s["description"], str) and len(s["description"]) > 40, f"scene {s['id']} description")
        check(all(re.fullmatch(r"group\d+", g) for g in s["faces"]), f"scene {s['id']} group names")

    # 2. timeline: tiles [0, true duration]; scene cuts are shot cuts
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=duration", "-of", "csv=p=0", str(ctx.video)],
                               capture_output=True, text=True).stdout.strip())
    for name in ("shots", "scenes"):
        xs = raw[name]
        check([x["id"] for x in xs] == list(range(len(xs))), f"{name} ids")
        check(xs[0]["start_seconds"] == 0 and abs(xs[-1]["end_seconds"] - dur) < 0.1,
              f"{name} span {xs[0]['start_seconds']}..{xs[-1]['end_seconds']} vs {dur}")
        check(all(a["end_seconds"] == b["start_seconds"] for a, b in zip(xs, xs[1:])), f"{name} not contiguous")
        check(all(x["end_seconds"] > x["start_seconds"] for x in xs), f"{name} empty interval")
    shot_starts = {s["start_seconds"] for s in raw["shots"]}
    check(all(s["start_seconds"] in shot_starts for s in raw["scenes"]), "scene start not on a shot boundary")

    # 3. faces: timecodes <-> files, folders <-> keys, descriptions only cite present groups
    faces_dir = out_dir / "faces"
    n_tc = 0
    for s in raw["scenes"]:
        sdir = faces_dir / f"scene_{s['id']:03d}"
        dirs = {p.name for p in sdir.iterdir() if p.is_dir()}
        check(dirs == set(s["faces"]), f"scene {s['id']} folders {sorted(dirs)} vs keys {sorted(s['faces'])}")
        for g, ts in s["faces"].items():
            n_tc += len(ts)
            check(ts == sorted(set(ts)), f"scene {s['id']} {g} timecodes not sorted/unique")
            check(all(s["start_seconds"] <= t < s["end_seconds"] for t in ts), f"scene {s['id']} {g} outside")
            files = sorted(p.name for p in (sdir / g).glob("*.jpg"))
            check(files == sorted(f"t{t:07.2f}.jpg" for t in ts), f"scene {s['id']} {g} files != timecodes")
        cited = set(re.findall(r"\bgroup\d+\b", s["description"]))
        check(cited <= set(s["faces"]), f"scene {s['id']} description cites absent {sorted(cited - set(s['faces']))}")
    all_jpgs = list(faces_dir.rglob("*.jpg"))
    check(len(all_jpgs) == n_tc, f"{len(all_jpgs)} crop files vs {n_tc} timecodes")
    bad = [p for p in all_jpgs if (im := cv2.imread(str(p))) is None or min(im.shape[:2]) < 16]
    check(not bad, f"{len(bad)} unreadable crops")

    # 4. report links
    report = out_dir / "qa" / "report.html"
    if report.exists():
        srcs = re.findall(r"src='([^']+)'", report.read_text())
        missing = [s for s in srcs if not (report.parent / s).exists()]
        check(not missing, f"report: {len(missing)} missing images")

    # 5. identity: re-embed every saved crop; is it nearest to its own folder's group?
    agree, total, per_group = _identity_check(ctx, all_jpgs)
    log.info("identity check: %d/%d re-embedded crops (%.1f%%) are nearest to their own group", agree, total,
             100 * agree / max(total, 1))
    worst = sorted(per_group.items(), key=lambda kv: kv[1][0] / kv[1][1])[:3]
    log.info("  lowest-agreement groups: %s", ", ".join(f"{g} {a}/{t}" for g, (a, t) in worst))
    check(total == 0 or agree / total >= 0.9, "identity agreement below 90%")

    if fails:
        for f in fails[:40]:
            log.error("verify: %s", f)
        raise SystemExit(f"cross-verification failed: {len(fails)} problem(s)")
    log.info("verify: OK: %d shots, %d scenes, %d face crops; schema, timeline, files, descriptions, report "
             "and identities all consistent", len(raw["shots"]), len(raw["scenes"]), n_tc)


def _identity_check(ctx: Ctx, jpgs: list[Path]):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                       providers=ctx.cfg["faces"]["providers"] or None)
    app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
    embs, labels = [], []
    for p in jpgs:
        img = cv2.imread(str(p))
        img = cv2.copyMakeBorder(img, 32, 32, 32, 32, cv2.BORDER_CONSTANT)  # room for the detector
        faces = app.get(img)
        if not faces:
            continue
        h, w = img.shape[:2]
        f = min(faces, key=lambda f: np.hypot((f.bbox[0] + f.bbox[2]) / 2 - w / 2, (f.bbox[1] + f.bbox[3]) / 2 - h / 2))
        embs.append(f.normed_embedding)
        labels.append(p.parent.name)
    E, labels = np.array(embs), np.array(labels)
    groups = sorted(set(labels))
    C = np.stack([E[labels == g].mean(0) for g in groups])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    pred = np.array(groups)[np.argmax(E @ C.T, axis=1)]
    per_group = {g: (int((pred[labels == g] == g).sum()), int((labels == g).sum())) for g in groups}
    return int((pred == labels).sum()), len(labels), per_group
