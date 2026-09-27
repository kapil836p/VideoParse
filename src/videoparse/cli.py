"""videoparse command line.

  videoparse run --url URL             full pipeline (stages whose outputs exist are skipped)
  videoparse run --force faces,scenes  re-run from the earliest forced stage onwards
  videoparse run --refine              also let Gemini refine the scene boundaries
  videoparse smoke                     check InsightFace + a 60 s Gemini call on Flash-Lite
  videoparse verify                    independent cross-verification of output/
"""

from __future__ import annotations

import argparse
import logging
import time

from .common import Ctx, log

# (stage, output that marks it done, runner)
STAGES = [
    ("download", "cache/meta.json"),
    ("shots", "cache/shots.json"),
    ("keyframes", "cache/keyframes.json"),
    ("transcript", "cache/transcript.json"),
    ("facedetect", "cache/face_dets.parquet"),
    ("faceclust", "cache/faces.parquet"),
    ("embed", "cache/shot_visual.npy"),
    ("scenes", "cache/scenes_local.json"),
    ("refine", "cache/refine.json"),
    ("describe", "cache/descriptions.json"),
    ("export", None),
    ("report", None),
]
ALIASES = {"faces": ["facedetect", "faceclust"]}


def _runner(name: str, ctx: Ctx, args):
    from . import describe, download, embed, export, faces, keyframes, report, scenes, shots, transcribe, validate

    return {
        "download": lambda: download.run(ctx, args.url),
        "shots": lambda: shots.run(ctx),
        "keyframes": lambda: keyframes.run(ctx),
        "transcript": lambda: transcribe.run(ctx),
        "facedetect": lambda: faces.detect(ctx),
        "faceclust": lambda: faces.cluster(ctx),
        "embed": lambda: embed.run(ctx),
        "scenes": lambda: scenes.run(ctx),
        "refine": lambda: describe.refine(ctx),
        "describe": lambda: describe.describe(ctx),
        "export": lambda: (export.run(ctx), validate.run(ctx)),
        "report": lambda: report.run(ctx),
    }[name]


def cmd_run(args) -> None:
    ctx = Ctx.load(args.root)
    forced = set()
    for f in filter(None, (args.force or "").split(",")):
        forced.update(ALIASES.get(f, [f]))
    unknown = forced - {s for s, _ in STAGES}
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}")
    dirty = False
    for name, marker in STAGES:
        if name == "refine" and not args.refine:
            continue
        done = marker is not None and (ctx.root / marker).exists()
        if name == "download" and not done and not args.url:
            raise SystemExit("--url is required for the first run")
        if done and not dirty and name not in forced:
            log.info("[%s] cached", name)
            continue
        dirty = True  # everything downstream of a re-run stage is re-run too
        t = time.time()
        log.info("[%s] running", name)
        _runner(name, ctx, args)()
        log.info("[%s] done in %.1fs", name, time.time() - t)


def cmd_smoke(args) -> None:
    import cv2
    from insightface.app import FaceAnalysis
    from pydantic import BaseModel

    from .gemini import Gemini

    ctx = Ctx.load(args.root)
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                       providers=ctx.cfg["faces"]["providers"] or None)
    app.prepare(ctx_id=0, det_size=(640, 640))
    for name, m in app.models.items():
        log.info("insightface %s providers: %s", name, m.session.get_providers())
    cap = cv2.VideoCapture(str(ctx.video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 60_000)
    ok, frame = cap.read()
    if ok:
        log.info("faces in frame @60s: %d", len(app.get(frame)))

    class Smoke(BaseModel):
        summary: str
        people_visible: int

    gem = Gemini(ctx)
    out, model = gem.ask([("video", (0.0, 60.0)),
                          ("text", "Summarise this clip in one sentence and count the people visible.")],
                         Smoke, models=gem.models[1:])
    log.info("gemini smoke (%s, source=%s): %s", model, gem.source, out)
    gem.cleanup()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(prog="videoparse")
    p.add_argument("--root", default=".")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url")
    r.add_argument("--force", help="comma-separated stages to re-run (faces = facedetect,faceclust)")
    r.add_argument("--refine", action="store_true", help="Gemini refinement of scene boundaries")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("smoke")
    s.set_defaults(fn=cmd_smoke)
    v = sub.add_parser("verify", help="independent cross-verification of output/")
    v.set_defaults(fn=lambda a: __import__("videoparse.crosscheck", fromlist=["run"]).run(Ctx.load(a.root)))
    args = p.parse_args()
    args.fn(args)
