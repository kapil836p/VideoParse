"""Light static HTML QA report: per scene time range, description, keyframes, one face per group."""

from __future__ import annotations

import html
import shutil

import cv2

from .common import Ctx, hms, log, read_json, resize_max

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--card:#f5f5f7;--line:#d2d2d7}
@media (prefers-color-scheme:dark){:root{--bg:#161617;--fg:#f5f5f7;--muted:#a1a1a6;--card:#232325;--line:#3a3a3c}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0 auto;
max-width:1100px;padding:24px 16px}
h1{font-size:24px;margin:0 0 4px}.muted{color:var(--muted)}
.scene{background:var(--card);border-radius:12px;padding:16px;margin:16px 0}
.scene h2{font-size:17px;margin:0 0 6px}.kf{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}
.kf img{height:96px;border-radius:6px}.faces{display:flex;gap:10px;flex-wrap:wrap}
.face{text-align:center;font-size:12px}.face img{width:72px;height:72px;object-fit:cover;border-radius:8px;display:block}
a{color:inherit}
"""


def run(ctx: Ctx) -> None:
    meta, out = ctx.meta(), read_json(ctx.output / "output.json")
    ext = {s["id"]: s for s in read_json(ctx.output / "output_extended.json")["scenes"]}
    kf_index = read_json(ctx.cache / "keyframes.json")
    qa = ctx.output / "qa"
    kf_dir = qa / "keyframes"
    shutil.rmtree(kf_dir, ignore_errors=True)
    kf_dir.mkdir(parents=True)

    parts = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>Scene Report</title><style>{CSS}</style>",
             f"<h1>{html.escape(meta['title'] or '')}</h1><p class=muted><a href='{html.escape(meta['url'])}'>"
             f"{html.escape(meta['url'])}</a> · {len(out['shots'])} shots · {len(out['scenes'])} scenes</p>"]
    for sc in out["scenes"]:
        e = ext[sc["id"]]
        first, last = e["shot_ids"]
        picks = sorted({first, (first + last) // 2, last})
        imgs = []
        for sid in picks:
            paths = kf_index.get(str(sid)) or []
            if not paths:
                continue
            img = cv2.imread(str(ctx.root / paths[len(paths) // 2]))
            name = f"scene_{sc['id']:03d}_shot_{sid:04d}.jpg"
            cv2.imwrite(str(kf_dir / name), resize_max(img, 320), [cv2.IMWRITE_JPEG_QUALITY, 80])
            imgs.append(f"<img src='keyframes/{name}' alt='shot {sid}'>")
        faces = []
        for g, ts in sc["faces"].items():
            thumb = f"../faces/scene_{sc['id']:03d}/{g}/t{ts[len(ts) // 2]:07.2f}.jpg"
            faces.append(f"<div class=face><a href='groups/{g}.jpg'><img src='{thumb}' alt='{g}'></a>"
                         f"{g}<br><span class=muted>{len(ts)} frames</span></div>")
        title = html.escape(e.get("title") or "")
        parts.append(
            f"<div class=scene><h2>Scene {sc['id']} · {hms(sc['start_seconds'])}–{hms(sc['end_seconds'])}"
            f" · {title}</h2><div class=muted>shots {first}–{last}</div>"
            f"<p>{html.escape(sc['description'])}</p><div class=kf>{''.join(imgs)}</div>"
            f"<div class=faces>{''.join(faces) or '<span class=muted>No faces</span>'}</div></div>")
    (qa / "report.html").write_text("\n".join(parts))
    log.info("report: %s", qa / "report.html")
