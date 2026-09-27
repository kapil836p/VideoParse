"""Stage 5: face detection, quality tiers, in-shot tracking, global identity clustering.

facedetect: sample frames on a fixed grid -> InsightFace buffalo_l -> crops + embeddings
faceclust:  tracks (Hungarian, within a shot) -> agglomerative clustering with cannot-link
            -> rescue -> group1..K by screen time -> contact sheets
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from .common import Ctx, iter_frames_at, log, shot_index, write_json


# --------------------------------------------------------------------------- detection

def sample_times(shots: list[dict], duration: float, fps: float, interval: float) -> list[float]:
    """Global grid 0, interval, 2*interval, ... plus the midpoint of shots that contain no grid point."""
    last = duration - 1.5 / fps
    grid = np.round(np.arange(0.0, last, interval), 2)
    times = set(grid.tolist())
    for s in shots:
        i = np.searchsorted(grid, s["start"], side="left")
        if i < len(grid) and grid[i] < s["end"]:
            continue
        mid = round(min((s["start"] + s["end"]) / 2, last), 2)
        if len(grid) == 0 or np.min(np.abs(grid - mid)) >= 0.1:
            times.add(mid)
    return sorted(times)


def detect(ctx: Ctx) -> None:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align

    cfg = ctx.cfg["faces"]
    meta, shots = ctx.meta(), ctx.shots()
    kwargs = {"providers": cfg["providers"]} if cfg.get("providers") else {}
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], **kwargs)
    app.prepare(ctx_id=0, det_thresh=cfg["det_thresh"], det_size=(cfg["det_size"], cfg["det_size"]))

    crops_dir = ctx.cache / "crops"
    crops_dir.mkdir(exist_ok=True)
    for old in crops_dir.glob("*.jpg"):
        old.unlink()

    times = sample_times(shots, meta["duration"], meta["fps"], cfg["sample_interval"])
    anchor, weak = cfg["anchor"], cfg["weak"]
    rows, embs = [], []
    for t, frame in tqdm(iter_frames_at(ctx.video, times, meta["fps"]), total=len(times), desc="faces"):
        H, W = frame.shape[:2]
        for f in app.get(frame):
            x1, y1, x2, y2 = (float(v) for v in f.bbox)
            size = min(x2 - x1, y2 - y1)
            score = float(f.det_score)
            if score < weak["min_score"] or size < weak["min_size"]:
                continue
            kps = np.asarray(f.kps, dtype=np.float32)
            aligned = face_align.norm_crop(frame, kps, 112)
            sharp = float(cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            eye_mid = (kps[0, 0] + kps[1, 0]) / 2
            yaw = float(abs(kps[2, 0] - eye_mid) / max(np.linalg.norm(kps[1] - kps[0]), 1.0))
            tier = "anchor" if (score >= anchor["min_score"] and size >= anchor["min_size"]
                                and sharp >= anchor["min_sharpness"] and yaw <= anchor["max_yaw_ratio"]) else "weak"
            det_id = len(rows)
            _save_crop(frame, (x1, y1, x2, y2), W, H, cfg, crops_dir / f"{det_id:06d}.jpg")
            rows.append({"det_id": det_id, "ts": round(t, 2), "shot_id": shot_index(shots, t),
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2, "det_score": score,
                         "sharpness": sharp, "yaw_ratio": yaw, "tier": tier})
            embs.append(np.asarray(f.normed_embedding, dtype=np.float32))

    df = pd.DataFrame(rows)
    df.to_parquet(ctx.cache / "face_dets.parquet", index=False)
    np.save(ctx.cache / "face_emb.npy", np.stack(embs) if embs else np.zeros((0, 512), np.float32))
    n_anchor = int((df["tier"] == "anchor").sum()) if len(df) else 0
    log.info("faces: %d sampled frames, %d detections (%d anchor, %d weak)",
             len(times), len(df), n_anchor, len(df) - n_anchor)


def _save_crop(frame, box, W, H, cfg, path) -> None:
    x1, y1, x2, y2 = box
    m = cfg["crop_margin"]
    w, h = x2 - x1, y2 - y1
    a, b = int(max(0, x1 - m * w)), int(max(0, y1 - m * h))
    c, d = int(min(W, x2 + m * w)), int(min(H, y2 + m * h))
    crop = frame[b:d, a:c]
    s = cfg["crop_max_side"] / max(crop.shape[:2])
    if s < 1:
        crop = cv2.resize(crop, (round(crop.shape[1] * s), round(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])


# --------------------------------------------------------------------------- tracking + clustering

def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (n,4) and b (m,4)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def build_tracks(df: pd.DataFrame, emb: np.ndarray, cfg: dict) -> np.ndarray:
    """Link detections of consecutive samples inside the same shot (Hungarian on 1 - IoU)."""
    track = np.full(len(df), -1, dtype=int)
    next_id = 0
    boxes = df[["x1", "y1", "x2", "y2"]].to_numpy()
    is_anchor = (df["tier"] == "anchor").to_numpy()
    for _, g in df.groupby("shot_id", sort=True):
        prev: np.ndarray | None = None
        for _, cur_df in g.groupby("ts", sort=True):
            cur = cur_df.index.to_numpy()
            if prev is not None and len(prev):
                iou = _iou(boxes[prev], boxes[cur])
                cos = emb[prev] @ emb[cur].T
                both_anchor = is_anchor[prev][:, None] & is_anchor[cur][None, :]
                ok = (iou >= cfg["min_iou"]) & (~both_anchor | (cos >= cfg["min_cos"]))
                cost = np.where(ok, 1 - iou, 10.0)
                for r, c in zip(*linear_sum_assignment(cost)):
                    if ok[r, c]:
                        track[cur[c]] = track[prev[r]]
            for i in cur:
                if track[i] < 0:
                    track[i] = next_id
                    next_id += 1
            prev = cur
    return track


def cluster(ctx: Ctx) -> None:
    cfg = ctx.cfg["faces"]
    df = pd.read_parquet(ctx.cache / "face_dets.parquet")
    emb = np.load(ctx.cache / "face_emb.npy")
    if df.empty:
        df["track_id"], df["group"] = [], []
        df.to_parquet(ctx.cache / "faces.parquet", index=False)
        write_json(ctx.cache / "face_groups.json", {})
        log.warning("no faces detected")
        return

    df["track_id"] = build_tracks(df, emb, cfg["track"])
    quality = (df["det_score"] * np.clip(df["sharpness"] / 200, 0.1, 1.0)).to_numpy()
    anchor = (df["tier"] == "anchor").to_numpy()

    # Track embeddings from anchor faces only.
    tids = sorted(df.loc[anchor, "track_id"].unique())
    T = np.zeros((len(tids), emb.shape[1]), np.float32)
    frames_of: list[set[float]] = []
    for k, tid in enumerate(tids):
        m = (df["track_id"] == tid).to_numpy()
        ma = m & anchor
        v = (emb[ma] * quality[ma, None]).sum(0)
        T[k] = v / (np.linalg.norm(v) + 1e-9)
        frames_of.append(set(df.loc[m, "ts"]))
    n_dets = df.groupby("track_id").size().reindex(tids).to_numpy()

    labels = _agglomerative(T, frames_of, cfg["cluster"]["distance_threshold"])
    labels = _fix_conflicts(T, frames_of, labels)

    # Dissolve tiny clusters (one track, few detections), then rescue leftovers.
    for lab in set(labels) - {-1}:
        members = np.where(labels == lab)[0]
        if len(members) == 1 and n_dets[members].sum() < cfg["cluster"]["min_detections"]:
            labels[members] = -1
    labels = _rescue(T, frames_of, labels, cfg["cluster"]["rescue_min_cos"])
    labels = _merge_close(T, frames_of, labels, cfg["cluster"]["merge_min_cos"],
                          df[anchor].groupby("track_id").size().reindex(tids).to_numpy())

    # Tracks seen only as weak faces (profile, blur) join a group only with strong evidence.
    track_label = dict(zip(tids, labels))
    track_label.update(_rescue_weak(df, emb, quality, T, tids, labels, frames_of,
                                    cfg["cluster"]["weak_rescue_min_cos"]))

    # group1..K by total detections (screen time), descending.
    df["cluster"] = df["track_id"].map(track_label).fillna(-1).astype(int)
    sizes = df[df["cluster"] >= 0].groupby("cluster").size().sort_values(ascending=False)
    rename = {c: i + 1 for i, c in enumerate(sizes.index)}
    df["group"] = df["cluster"].map(rename).fillna(-1).astype(int)
    df = df.drop(columns="cluster")
    df.to_parquet(ctx.cache / "faces.parquet", index=False)

    stats = _stats(df, emb)
    write_json(ctx.cache / "face_groups.json", stats)
    _contact_sheets(ctx, df)
    kept = df[df["group"] > 0]
    log.info("face groups: %d groups from %d tracks; kept %d/%d detections (dropped %d)",
             len(rename), len(tids), len(kept), len(df), len(df) - len(kept))
    for g, s in stats["groups"].items():
        log.info("  %-8s dets=%4d tracks=%3d intra_cos=%.2f  nearest=%s (%.2f)", g, s["detections"],
                 s["tracks"], s["mean_cos_to_centroid"], s["nearest_group"], s["nearest_cos"])


def _agglomerative(T: np.ndarray, frames_of: list[set], thr: float) -> np.ndarray:
    n = len(T)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    D = np.clip(1 - T @ T.T, 0, 2)
    for i in range(n):
        for j in range(i + 1, n):
            if frames_of[i] & frames_of[j]:  # seen in the same frame -> different people
                D[i, j] = D[j, i] = 2.0
    np.fill_diagonal(D, 0)
    return AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                   distance_threshold=thr).fit(D).labels_.astype(int)


def _centroid(T, idx):
    v = T[idx].mean(0)
    return v / (np.linalg.norm(v) + 1e-9)


def _fix_conflicts(T, frames_of, labels):
    """If a cluster still holds two tracks sharing a frame, keep the one closer to the centroid."""
    labels = labels.copy()
    for lab in set(labels) - {-1}:
        members = list(np.where(labels == lab)[0])
        c = _centroid(T, members)
        members.sort(key=lambda i: -float(T[i] @ c))
        seen: set = set()
        for i in members:
            if frames_of[i] & seen:
                labels[i] = -1
            else:
                seen |= frames_of[i]
    return labels


def _rescue(T, frames_of, labels, min_cos):
    labels = labels.copy()
    labs = sorted(set(labels) - {-1})
    for i in np.where(labels == -1)[0]:
        best, best_cos = -1, min_cos
        for lab in labs:
            members = np.where(labels == lab)[0]
            if any(frames_of[i] & frames_of[j] for j in members):
                continue
            cos = float(T[i] @ _centroid(T, members))
            if cos >= best_cos:
                best, best_cos = lab, cos
        labels[i] = best
    return labels


def _rescue_weak(df, emb, quality, T, tids, labels, frames_of, min_cos) -> dict:
    """Assign weak-only tracks to the nearest identity if cosine >= min_cos and no frame conflict."""
    labs = sorted(set(labels) - {-1})
    if not labs:
        return {}
    members = {lab: np.where(labels == lab)[0] for lab in labs}
    cents = np.stack([_centroid(T, members[lab]) for lab in labs])
    frames = {lab: set().union(*(frames_of[i] for i in members[lab])) for lab in labs}
    anchored = set(tids)
    cands = []
    for tid, t in df.groupby("track_id"):
        if tid in anchored:
            continue
        idx = t.index.to_numpy()
        v = (emb[idx] * quality[idx, None]).sum(0)
        s = cents @ (v / (np.linalg.norm(v) + 1e-9))
        cands += [(float(s[j]), tid, labs[j], set(t["ts"])) for j in np.where(s >= min_cos)[0]]
    out: dict = {}
    for _, tid, lab, ts in sorted(cands, key=lambda c: -c[0]):
        if tid not in out and not ts & frames[lab]:
            out[tid] = lab
            frames[lab] |= ts
    return out


def _merge_close(T, frames_of, labels, min_cos, weights):
    """Merge clusters of the same person split by lighting/pose: centroid cosine >= min_cos and
    never on screen in the same frame. Most similar pair first, centroids recomputed after each merge."""
    labels = labels.copy()
    while True:
        labs = sorted(set(labels) - {-1})
        cents, frames = {}, {}
        for lab in labs:
            idx = np.where(labels == lab)[0]
            v = (T[idx] * weights[idx, None]).sum(0)
            cents[lab] = v / (np.linalg.norm(v) + 1e-9)
            frames[lab] = set().union(*(frames_of[i] for i in idx))
        best = None
        for i, a in enumerate(labs):
            for b in labs[i + 1:]:
                cos = float(cents[a] @ cents[b])
                if cos >= min_cos and not frames[a] & frames[b] and (best is None or cos > best[0]):
                    best = (cos, a, b)
        if best is None:
            return labels
        log.info("  merging clusters %d + %d (centroid cos %.2f)", best[1], best[2], best[0])
        labels[labels == best[2]] = best[1]


def _stats(df: pd.DataFrame, emb: np.ndarray) -> dict:
    kept = df[(df["group"] > 0) & (df["tier"] == "anchor")]
    cents = {}
    out = {}
    for g, gd in kept.groupby("group"):
        E = emb[gd["det_id"].to_numpy()]
        c = E.mean(0)
        c /= np.linalg.norm(c) + 1e-9
        cents[g] = c
        out[f"group{g}"] = {"detections": int((df["group"] == g).sum()),
                            "tracks": int(df.loc[df["group"] == g, "track_id"].nunique()),
                            "mean_cos_to_centroid": float((E @ c).mean())}
    for g, c in cents.items():
        others = [(float(c @ c2), g2) for g2, c2 in cents.items() if g2 != g]
        cos, g2 = max(others) if others else (0.0, None)
        out[f"group{g}"].update(nearest_group=f"group{g2}" if g2 else None, nearest_cos=cos)
    return {"groups": out}


def _contact_sheets(ctx: Ctx, df: pd.DataFrame, n: int = 64, tile: int = 96) -> None:
    out = ctx.output / "qa" / "groups"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for g, gd in df[df["group"] > 0].groupby("group"):
        gd = gd.sort_values("ts")
        pick = gd.iloc[np.linspace(0, len(gd) - 1, min(n, len(gd))).round().astype(int)]
        cols = 8
        rows = int(np.ceil(len(pick) / cols))
        sheet = np.full((rows * tile, cols * tile, 3), 255, np.uint8)
        for k, det_id in enumerate(pick["det_id"]):
            img = cv2.imread(str(ctx.cache / "crops" / f"{det_id:06d}.jpg"))
            if img is None:
                continue
            img = cv2.resize(img, (tile, tile))
            r, c = divmod(k, cols)
            sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = img
        cv2.imwrite(str(out / f"group{g}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def best_crops(df: pd.DataFrame, group: int, k: int, t0: float = 0, t1: float = 1e9) -> list[int]:
    """det_ids of the k best anchor crops of a group in [t0, t1), spread over time."""
    gd = df[(df["group"] == group) & (df["ts"] >= t0) & (df["ts"] < t1)]
    a = gd[gd["tier"] == "anchor"]
    gd = a if len(a) else gd
    if gd.empty:
        return []
    gd = gd.assign(q=gd["det_score"] * np.clip(gd["sharpness"] / 200, 0.1, 1.0))
    top = gd.nlargest(min(len(gd), k * 4), "q").sort_values("ts")
    idx = np.linspace(0, len(top) - 1, min(k, len(top))).round().astype(int)
    return top.iloc[idx]["det_id"].tolist()


def group_signature(ctx: Ctx) -> str:
    """Stable hash of the detection->group mapping (part of Gemini cache keys)."""
    import hashlib

    df = pd.read_parquet(ctx.cache / "faces.parquet", columns=["det_id", "group"])
    return hashlib.sha256(json.dumps(df.values.tolist()).encode()).hexdigest()[:16]
