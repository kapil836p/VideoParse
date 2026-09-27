"""Stage 4: speech-to-text with timestamps (parakeet-mlx -> mlx-whisper -> YouTube captions)."""

from __future__ import annotations

from .common import Ctx, log, read_json, write_json


def run(ctx: Ctx) -> None:
    cfg = ctx.cfg["transcript"]
    for name, fn in (("parakeet", _parakeet), ("whisper", _whisper), ("youtube_captions", _captions)):
        try:
            segments = fn(ctx, cfg)
        except Exception as e:  # noqa: BLE001 - try the next backend
            log.warning("transcript backend %s failed: %s", name, e)
            continue
        if segments:
            write_json(ctx.cache / "transcript.json", {"source": name, "segments": segments})
            log.info("transcript (%s): %d segments, %d words", name, len(segments),
                     sum(len(s["text"].split()) for s in segments))
            return
    raise RuntimeError("all transcript backends failed")


def _parakeet(ctx: Ctx, cfg: dict) -> list[dict]:
    from parakeet_mlx import from_pretrained

    model = from_pretrained(cfg["parakeet_model"])
    res = model.transcribe(str(ctx.audio), chunk_duration=cfg["chunk_seconds"], overlap_duration=15.0)
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in res.sentences if s.text.strip()]


def _whisper(ctx: Ctx, cfg: dict) -> list[dict]:
    import mlx_whisper

    res = mlx_whisper.transcribe(str(ctx.audio), path_or_hf_repo=cfg["whisper_model"])
    return [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
            for s in res["segments"] if s["text"].strip()]


def _captions(ctx: Ctx, cfg: dict) -> list[dict]:
    files = sorted(ctx.data.glob("video.en*.json3"))
    if not files:
        return []
    segs = []
    for ev in read_json(files[0]).get("events", []):
        text = "".join(s.get("utf8", "") for s in ev.get("segs", [])).replace("\n", " ").strip()
        if text:
            start = ev["tStartMs"] / 1000
            segs.append({"start": round(start, 2), "end": round(start + ev.get("dDurationMs", 0) / 1000, 2),
                         "text": text})
    return segs
