"""Gemini free-tier client: disk cache, token-per-minute pacing, retries, model + transport fallback.

A request is a list of logical items:
    ("text", str) | ("image", jpeg_bytes) | ("video", (start_s, end_s))
Videos are sent as the YouTube URL with clip offsets; if the API rejects that, the window is cut
locally to 360p and uploaded via the Files API instead. The cache key is computed from the logical
items (not the transport), so re-runs never spend quota.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections import deque

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from .common import Ctx, log


class QuotaExhausted(Exception):
    pass


class Gemini:
    def __init__(self, ctx: Ctx):
        load_dotenv(ctx.root / ".env")
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing: create a free key at https://aistudio.google.com/apikey "
                               "and put it in .env")
        self.ctx = ctx
        self.cfg = ctx.cfg["gemini"]
        self.client = genai.Client(api_key=key)
        self.models = [os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
                       os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")]
        self.source = os.environ.get("GEMINI_VIDEO_SOURCE", "youtube")  # or "upload"
        self.url = ctx.meta()["url"]
        self.cache_dir = ctx.cache / "gemini"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._window: deque[tuple[float, int]] = deque()
        self._uploads: dict[tuple, types.File] = {}

    # ------------------------------------------------------------------ public
    def ask(self, items: list[tuple], schema: type[BaseModel], models: list[str] | None = None) -> tuple[BaseModel, str]:
        models = models or self.models
        keys = {m: self._key(m, items, schema) for m in models}
        for m in models:  # any cached answer wins
            hit = self.cache_dir / f"{keys[m]}.json"
            if hit.exists():
                return schema.model_validate_json(json.loads(hit.read_text())["text"]), m
        last: Exception | None = None
        for m in models:
            try:
                text, usage = self._call(m, items, schema)
            except QuotaExhausted as e:
                log.warning("daily quota exhausted for %s; trying next model", m)
                last = e
                continue
            except errors.APIError as e:
                if e.code != 404:
                    raise
                log.warning("model %s not available (%s); trying next model", m, str(e)[:120])
                last = e
                continue
            (self.cache_dir / f"{keys[m]}.json").write_text(
                json.dumps({"model": m, "text": text, "usage": usage}, indent=2))
            return schema.model_validate_json(text), m
        raise RuntimeError(f"all Gemini models failed: {last}")

    # ------------------------------------------------------------------ internals
    def _key(self, model: str, items: list[tuple], schema: type[BaseModel]) -> str:
        logical = []
        for kind, val in items:
            if kind == "image":
                val = hashlib.sha256(val).hexdigest()
            elif kind == "video":
                val = [self.url, round(val[0], 2), round(val[1], 2)]
            logical.append([kind, val])
        blob = json.dumps({"model": model, "items": logical, "schema": schema.model_json_schema(),
                           "media_resolution": self.cfg["media_resolution"]}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _parts(self, items: list[tuple]) -> list[types.Part]:
        parts = []
        for kind, val in items:
            if kind == "text":
                parts.append(types.Part.from_text(text=val))
            elif kind == "image":
                parts.append(types.Part.from_bytes(data=val, mime_type="image/jpeg"))
            elif kind == "video":
                s, e = val
                if self.source == "youtube":
                    parts.append(types.Part(
                        file_data=types.FileData(file_uri=self.url),
                        video_metadata=types.VideoMetadata(start_offset=f"{int(s)}s", end_offset=f"{int(e + 0.999)}s")))
                else:
                    f = self._upload(s, e)
                    parts.append(types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type)))
        return parts

    def _upload(self, s: float, e: float) -> types.File:
        if (s, e) in self._uploads:
            return self._uploads[(s, e)]
        clip = self.ctx.cache / "clips" / f"clip_{s:.0f}_{e:.0f}.mp4"
        clip.parent.mkdir(exist_ok=True)
        if not clip.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                            "-i", str(self.ctx.video), "-vf", "scale=-2:360", "-c:v", "libx264", "-preset",
                            "veryfast", "-crf", "30", "-c:a", "aac", "-b:a", "64k", str(clip)], check=True)
        f = self.client.files.upload(file=str(clip))
        while f.state and f.state.name == "PROCESSING":
            time.sleep(3)
            f = self.client.files.get(name=f.name)
        self._uploads[(s, e)] = f
        return f

    def _estimate(self, items: list[tuple]) -> int:
        n = 0
        for kind, val in items:
            n += {"text": lambda v: len(v) // 3, "image": lambda v: 300,
                  "video": lambda v: int((v[1] - v[0]) * 110)}[kind](val)
        return n

    def _pace(self, tokens: int) -> None:
        budget = self.cfg["tpm_budget"]
        if tokens > budget:
            raise ValueError(f"request needs ~{tokens} tokens > per-minute budget {budget}; shorten the window")
        while True:
            now = time.time()
            while self._window and now - self._window[0][0] > 60:
                self._window.popleft()
            used = sum(t for _, t in self._window)
            if used + tokens <= budget:
                break
            wait = 61 - (now - self._window[0][0])
            log.info("pacing: %d tokens used in the last minute, waiting %.0fs", used, wait)
            time.sleep(max(wait, 1))
        self._window.append((time.time(), tokens))

    def _call(self, model: str, items: list[tuple], schema: type[BaseModel]) -> tuple[str, dict]:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            media_resolution=getattr(types.MediaResolution, f"MEDIA_RESOLUTION_{self.cfg['media_resolution']}"),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        switched = False
        for attempt in range(6):
            parts = self._parts(items)
            contents = [types.Content(role="user", parts=parts)]
            try:
                tokens = self.client.models.count_tokens(model=model, contents=contents).total_tokens
            except Exception:  # noqa: BLE001 - counting is best-effort
                tokens = self._estimate(items)
            self._pace(tokens)
            try:
                log.info("gemini %s: ~%d input tokens (source=%s)", model, tokens, self.source)
                resp = self.client.models.generate_content(model=model, contents=contents, config=config)
                usage = resp.usage_metadata.model_dump(mode="json") if resp.usage_metadata else {}
                if not resp.text:
                    raise RuntimeError(f"empty response: {resp.candidates}")
                return resp.text, usage
            except errors.APIError as e:
                msg = str(e)
                if e.code == 429 and re.search(r"per.?day|PerDay|daily", msg, re.I):
                    raise QuotaExhausted(msg) from e
                if e.code == 400 and self.source == "youtube" and not switched and "video" in msg.lower():
                    log.warning("YouTube URL input rejected (%s); switching to Files API upload", msg[:200])
                    self.source, switched = "upload", True
                    continue
                if e.code in (429, 500, 502, 503, 504):
                    m = re.search(r"retry(?:Delay)?['\"]?:?\s*['\"]?(\d+)", msg)
                    wait = int(m.group(1)) + 1 if m else min(10 * 2 ** attempt, 90)
                    log.warning("gemini %s error %s; retrying in %ss", model, e.code, wait)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"gemini {model}: too many retries")

    def cleanup(self) -> None:
        for f in self._uploads.values():
            try:
                self.client.files.delete(name=f.name)
            except Exception:  # noqa: BLE001
                pass
