# videoparse

Turns a 25–30 minute YouTube video into:
- **shots**: every camera cut
- **thematic scenes**: consecutive shots that share a theme
- **a description per scene**: central theme, key contributors, events
- **face groups**: detected faces clustered by identity, saved as one folder per group per scene

Everything runs locally on a Mac (Apple Silicon) at **$0**. The only exception is the scene descriptions, which use the free tier of the Gemini API.

## Submission

| Item | Location |
|---|---|
| Video | *Younger* – Season 7 Episode 5 (Full Episode), 26:38 – https://youtu.be/Y57MApUlDhk |
| Output JSON (exact assignment schema) | [`output/output.json`](output/output.json) |
| Extended JSON (titles, shot ranges, contributors, events, HH:MM:SS timecodes, appearance intervals) | [`output/output_extended.json`](output/output_extended.json) |
| Faces folder tree | [`output/faces_tree.txt`](output/faces_tree.txt) |
| Face crops | `output/faces/scene_XXX/groupN/tSSSS.ss.jpg` |
| Visual QA report | [`output/qa/report.html`](output/qa/report.html), plus per-group contact sheets in `output/qa/groups/` |
| Code | `src/videoparse/` |

## Output conventions
- **Shots and scenes tile the video.** Shot 0 starts at 0.0, each `end_seconds` equals the next `start_seconds`, and the last one ends at the video duration.
- **Scenes are contiguous runs of whole shots,** so every scene boundary is also a shot boundary.
- **Face groups are global identities.** `group3` is the same person in every scene. Groups are numbered by total screen time, so `group1` is the most-seen person. A scene lists only the groups that appear in it.
- **Timecodes** are seconds, taken from frames sampled once per second, plus one frame for any shot shorter than that.
  - Every timecode in `faces` has exactly one crop file named after it. For example, `"group2": [452.0, …]` ↔ `faces/scene_007/group2/t0452.00.jpg`.
  - A scene without faces has `"faces": {}` and a folder containing only a `_no_faces` marker.
- **Descriptions** follow the template's structure: `Central theme: … Key contributors: group1 (host, named '…' per on-screen caption), … Events: (1) …; (2) …`.

## How to run
```bash
brew install ffmpeg deno uv tree
uv sync
cp .env.example .env          # add a free Gemini key from https://aistudio.google.com/apikey
uv run videoparse run --url "https://www.youtube.com/watch?v=Y57MApUlDhk" --refine
uv run videoparse verify      # independent cross-check of everything in output/
```
- **Caching:** each stage caches its result in `cache/`. Re-running skips finished stages, and `--force faces,scenes` re-runs from the earliest forced stage onwards.
- **Refinement:** `--refine` additionally lets Gemini choose among the candidate scene boundaries.
- **Smoke test:** `uv run videoparse smoke` checks the face models and makes one 60-second Gemini call on Flash-Lite.
- **Settings:** all thresholds live in [`config.yaml`](config.yaml).

## Pipeline and model choices

| Stage | Tool / model | Why |
|---|---|---|
| Download | `yt-dlp` (+ deno for YouTube's JS challenges), 720p H.264 | Reliable. 720p is enough for face crops and fast to decode. |
| Shots | **TransNetV2** (`transnetv2-pytorch`, CPU); PySceneDetect `AdaptiveDetector` as fallback | A neural shot-boundary detector that also catches dissolves and fades. Runs on CPU because MPS gives run-to-run differences. |
| Transcript | **Parakeet TDT 0.6B v3** via `parakeet-mlx`; fallbacks are Whisper large-v3-turbo (`mlx-whisper`) and YouTube captions | ~6.3% average WER, sentence timestamps, very fast on Apple Silicon. |
| Visual similarity | **SigLIP2** ViT-B/16 (`open_clip`) | Keyframe embeddings capture "same set / same camera setup". |
| Topic similarity | **bge-small-en-v1.5** (`sentence-transformers`) | Small, strong sentence embeddings. |
| Faces | **InsightFace `buffalo_l`**: SCRFD-10GF detector + ArcFace R50 (512-d) | State-of-the-art open face detection and recognition. |
| Face grouping | In-shot tracking (Hungarian on IoU), then agglomerative clustering (average linkage, cosine distance 0.55) with a cannot-link rule for faces in the same frame, then a merge pass for identities split by lighting | Identities are unknown in advance. Tracks remove near-duplicate frames, and cannot-link keeps two people who appear together from being merged. |
| Scene segmentation | Multimodal **TextTiling** over shot boundaries, then **Gemini** picks among the candidate cuts | See below. |
| Descriptions | **Gemini** (newest free Flash model, `gemini-3.8-flash`; Flash-Lite as fallback), reading the **YouTube URL directly**, with a JSON schema | Gemini watches the frames *and* hears the audio. It gets the scene's transcript and a gallery of each face group, so it can name contributors by group label. |

### Scene segmentation ("common theming")
A scene here is a stretch of shots in one location and time, with one set of people in one conversation or activity (one theme).

**1. Local multimodal scoring.** At every shot boundary, the pipeline compares what comes before with what comes after on three signals:
- **Setting / camera setup:** the best-matching pair of SigLIP2 shot embeddings, 6 shots on each side of the cut. Inside a scene the editor keeps returning to the same camera setups (shot / reverse shot). Across a scene change, no shot after the cut resembles one before it. This "shot-link" idea comes from logical-story-unit detection, and it was much sharper here than comparing average embeddings.
- **Topic:** transcript embeddings for 60 s on each side.
- **People:** screen time per face group for 60 s on each side.

Each signal is z-scored, missing signals (silence, no faces) are skipped, and the rest are combined with weights 0.35 / 0.45 / 0.20. A TextTiling *depth score* measures how deep the similarity valley is at each boundary.

**2. Choosing cuts.**
- Cuts are taken deepest-first, keeping every scene at least 30 s long.
- YouTube's auto-generated chapters only add a small bonus; they turned out to be a few seconds off the real scene changes.
- The 60 deepest boundaries (at least 15 s apart) become candidates.

**3. Gemini refinement (`--refine`).** Gemini watches the whole episode (one request, ~176K tokens at LOW media resolution) and chooses which candidates start a new scene. It can only choose among exact shot boundaries, so no imprecise LLM timestamps reach the JSON.

**4. Establishing shots.** This show opens scenes with faceless exterior shots (skylines, street signs). Each cut is moved back over up to 4 such shots (≤ 15 s), so they open the scene they introduce. A shot only moves if it looks more like the following shots than the preceding ones, so a closing shot (e.g. the wall two characters just walked past) stays with its own scene.

### Face pipeline details
- **Two quality tiers:**
  - **Anchor** faces build identities: score ≥ 0.65, ≥ 48 px, sharp, near-frontal.
  - **Weak** faces (profile, blurry, small) inherit the identity of the track they belong to.
- **Weak-only tracks**, where someone is only ever seen in profile, join a group only when their similarity to it is ≥ 0.55. That's stricter than the 0.45 used to build groups; the band just above 0.55 was checked by hand at about 90% precision. Everything else (background extras, over-the-shoulder shots, backs of heads) is left out.
- **Merge pass:** after clustering, groups whose centroids have cosine ≥ 0.30 *and* that never appear in the same frame are merged.
  - On this video, same-person splits caused by lighting (e.g. purple party light, dim restaurant) measured 0.32–0.66, while different people measured ≤ 0.20.
  - The merge fixed 6 split identities.
- Tiny clusters (one track, fewer than 3 frames) are dropped as noise.
- The per-group contact sheets in `output/qa/groups/` were used to tune these thresholds.

### Gemini free-tier handling
- Requests go one at a time, with `count_tokens` and a rolling tokens-per-minute budget (230K).
- Media resolution is set to LOW (~100 tokens per second of video).
- Scenes are batched into ~10-minute windows (about 3 requests per video).
- Retries honour the API's retry delay, and the pipeline falls back from Flash to Flash-Lite on daily-quota errors.
- If the API rejects clipping of YouTube URLs, the pipeline falls back to uploading a 360p clip through the Files API.
- Every response is cached in `cache/gemini/`, keyed by model, prompt, images, video window and face grouping, so re-runs cost no quota.
- **Consistent names:** a person named in one scene (from dialogue or on-screen text) keeps that name in every scene, by majority vote per face group. The final description strings are composed deterministically from Gemini's structured fields.
- **Requests used for this video:** 8 in total. One Flash-Lite smoke test, three whole-video refinement attempts while tuning the prompt and candidates, and four description requests (three windows, plus one window re-described after a boundary fix).

## Results

| | Value |
|---|---|
| Video duration | 1,597.6 s (26:38), 720p at 25 fps |
| Shots | 546 (median length 2.3 s) |
| Scenes | 19 (34 s – 4:47, median 74 s) |
| Face groups | 20 (13 named by Gemini from dialogue or on-screen evidence) |
| Face crops / timecodes | 1,752 (1 per second of screen time per person) |
| Output size | 19 MB |

Face groups and the character names Gemini attached to them (see `output/qa/groups/`):

| Group | Character | | Group | Character |
|---|---|---|---|---|
| group1 | Liza | | group8 | Josh |
| group2 | Vince | | group9 | Michelle |
| group3 | Kelsey | | group10 | Claire |
| group4 | Lauren | | group11 | Camilla |
| group5 | Maggie | | group14 | Ayana Williams |
| group6 | Charles | | group15 | Aiden |
| group7 | Quinn Tyler | | others | unnamed minor characters and a baby |

### Evaluation
- **Automated checks:** `videoparse verify` re-derives everything from `output/` and ffprobe, without using any pipeline state, and all checks pass:
  - the exact JSON schema
  - shots and scenes tile [0, duration] with no gaps
  - every scene starts on a shot boundary
  - a 1-to-1 match between timecodes and crop files
  - descriptions only cite groups present in their scene
  - every image in the report exists
- **Face identity:** every saved crop was re-embedded from scratch. **99.7%** (1,746 of 1,751) are nearest to their own group's centroid.
- **Face groups (manual):** every group's contact sheet was inspected, and all 20 contain a single person. Gemini's names are consistent across three independent requests (e.g. group1 = "Liza" in 12 of 12 mentions).
- **Shot boundaries (manual):** 20 randomly sampled cuts, frame before vs. frame after: **20/20** are real cuts.
- **Scene boundaries (manual):** the frames on each side of all 18 final scene cuts were reviewed.
  - 15 are changes of location or time, usually marked by an establishing exterior shot.
  - 3 (4:51, 9:36, 11:09) stay in the same place (office, party) but switch to a different conversation.
  - Problems found and fixed while iterating: chapter-forced cuts a few seconds off the real change, establishing shots attached to the wrong scene, and one closing shot that was moved wrongly.

## Limitations
- **Faces on screens or in photos** inside the video (e.g. a picture held up, a TV in the background) can join the real person's group or form a small group of their own.
- **Very similar-looking people** (e.g. siblings), or strong changes in make-up or lighting, can merge or split groups. The threshold is a trade-off.
- **Scene boundaries** are a judgement call. The algorithm follows topic, setting and participant changes, so a scene may differ from how a human editor would split the video.
- **Descriptions** are generated by an LLM. They were checked manually against the video, but may contain small inaccuracies.

## Licences and terms
- **InsightFace model weights** (`buffalo_l`) are licensed for **non-commercial research use only**. This project is an academic assignment.
- **The video** is downloaded only for this research and educational analysis. Copyright remains with its owner.
- **Gemini free tier:** Google may use free-tier prompts to improve its products. Only a public YouTube video and derived data were sent.
