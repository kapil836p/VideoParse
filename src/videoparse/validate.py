"""Integrity checks on output/output.json and output/faces (runs after every export)."""

from __future__ import annotations

from pydantic import BaseModel

from .common import Ctx, log, read_json


class Shot(BaseModel):
    id: int
    start_seconds: float
    end_seconds: float


class Scene(Shot):
    description: str
    faces: dict[str, list[float]]


class Output(BaseModel):
    shots: list[Shot]
    scenes: list[Scene]


def run(ctx: Ctx) -> None:
    meta = ctx.meta()
    raw = read_json(ctx.output / "output.json")
    out = Output.model_validate(raw)
    errors, warnings = [], []
    tol = 1.5 / meta["fps"]
    duration = meta["duration"]

    if set(raw) != {"shots", "scenes"}:
        errors.append(f"top-level keys must be shots, scenes: {sorted(raw)}")
    for name, items in (("shots", out.shots), ("scenes", out.scenes)):
        if [x.id for x in items] != list(range(len(items))):
            errors.append(f"{name}: ids are not 0..n-1")
        if not items or abs(items[0].start_seconds) > tol or abs(items[-1].end_seconds - duration) > tol:
            errors.append(f"{name}: do not cover [0, {duration:.3f}]")
        for a, b in zip(items, items[1:]):
            if abs(a.end_seconds - b.start_seconds) > 1e-6:
                errors.append(f"{name}: gap/overlap between {a.id} and {b.id}")
        for x in items:
            if x.end_seconds <= x.start_seconds:
                errors.append(f"{name}: {x.id} has non-positive length")

    shot_bounds = {round(s.start_seconds, 3) for s in out.shots}
    min_len = ctx.cfg["scenes"]["min_scene_seconds"]
    faces_dir = ctx.output / "faces"
    for sc in out.scenes:
        if round(sc.start_seconds, 3) not in shot_bounds:
            errors.append(f"scene {sc.id} does not start on a shot boundary")
        if sc.end_seconds - sc.start_seconds < min_len:
            warnings.append(f"scene {sc.id} is shorter than {min_len}s")
        if not sc.description.strip():
            errors.append(f"scene {sc.id} has no description")
        sdir = faces_dir / f"scene_{sc.id:03d}"
        if not sdir.is_dir():
            errors.append(f"missing folder {sdir.name}")
            continue
        folders = {p.name for p in sdir.iterdir() if p.is_dir()}
        if folders != set(sc.faces):
            errors.append(f"scene {sc.id}: folders {sorted(folders)} != face keys {sorted(sc.faces)}")
        if not sc.faces and not (sdir / "_no_faces").exists():
            errors.append(f"scene {sc.id}: empty scene without _no_faces marker")
        for g, ts in sc.faces.items():
            if len(ts) != len(set(ts)):
                errors.append(f"scene {sc.id} {g}: duplicate timecodes")
            if any(not (sc.start_seconds <= t < sc.end_seconds + 1e-6) for t in ts):
                errors.append(f"scene {sc.id} {g}: timecode outside scene")
            files = {p.name for p in (sdir / g).glob("*.jpg")} if (sdir / g).is_dir() else set()
            expected = {f"t{t:07.2f}.jpg" for t in ts}
            if files != expected:
                errors.append(f"scene {sc.id} {g}: {len(files ^ expected)} files do not match timecodes")

    for w in warnings:
        log.warning("validate: %s", w)
    if errors:
        for e in errors[:50]:
            log.error("validate: %s", e)
        raise SystemExit(f"validation failed with {len(errors)} error(s)")
    n_faces = sum(len(t) for s in out.scenes for t in s.faces.values())
    log.info("validate: OK (%d shots, %d scenes, %d face timecodes, %d warnings)", len(out.shots),
             len(out.scenes), n_faces, len(warnings))
