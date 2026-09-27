"""Stage 1: download the video with yt-dlp, probe it, extract 16 kHz mono audio."""

from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction

from .common import Ctx, log, read_json, write_json


def run(ctx: Ctx, url: str) -> None:
    out_tmpl = str(ctx.data / "video.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", ctx.cfg["download"]["format"],
        "--merge-output-format", "mp4",
        "--write-subs", "--write-auto-subs",
        "--sub-langs", "en.*,-live_chat",
        "--sub-format", "json3/vtt",
        "--write-info-json",
        "--no-playlist",
        "-o", out_tmpl,
        url,
    ]
    log.info("yt-dlp: %s", url)
    subprocess.run(cmd, check=True)
    if not ctx.video.exists():
        raise FileNotFoundError(f"yt-dlp did not produce {ctx.video}")

    info = read_json(ctx.data / "video.info.json")
    probe = ffprobe(ctx)
    meta = {
        "url": url,
        "video_id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "youtube_duration": info.get("duration"),
        "chapters": [
            {"title": c.get("title"), "start": c.get("start_time"), "end": c.get("end_time")}
            for c in (info.get("chapters") or [])
        ],
        **probe,
    }
    write_json(ctx.cache / "meta.json", meta)
    log.info("video: %s | %.1fs @ %.3f fps, %dx%d", meta["title"], meta["duration"], meta["fps"],
             meta["width"], meta["height"])

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(ctx.video), "-vn", "-ac", "1", "-ar", "16000",
         str(ctx.audio)],
        check=True,
    )


def ffprobe(ctx: Ctx) -> dict:
    """Exact video-stream duration and frame rate (not the container's)."""
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=duration,r_frame_rate,avg_frame_rate,nb_read_packets,width,height",
         "-show_entries", "format=duration", "-of", "json", str(ctx.video)],
        check=True, capture_output=True, text=True,
    )
    j = json.loads(res.stdout)
    st = j["streams"][0]
    fps = float(Fraction(st.get("avg_frame_rate") or st["r_frame_rate"]))
    n_frames = int(st.get("nb_read_packets") or 0)
    duration = float(st.get("duration") or j["format"]["duration"])
    return {"duration": duration, "fps": fps, "n_frames": n_frames,
            "width": int(st["width"]), "height": int(st["height"])}
