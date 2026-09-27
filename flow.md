# Flow

What happens from the moment you give it a YouTube link to the final JSON.

Command I used:

```
uv run videoparse run --url "https://youtu.be/Y57MApUlDhk" --refine
```

Every step saves its result in `cache/`, so if something breaks you can re-run and it continues from where it stopped.


## Steps

**1. Download**
yt-dlp downloads the video (720p mp4), the YouTube subtitles and the video info (title, chapters).
ffmpeg pulls out the audio as a wav file, and ffprobe gives the exact duration and fps.

**2. Shots**
TransNetV2 goes through every frame and gives the chance of a cut at that frame. Anything above 0.5 is a cut.
Very tiny shots (under 0.4s) get merged into the next one only if they look the same.
If TransNetV2 fails, PySceneDetect is used instead.

**3. Keyframes**
Grab 3 frames from each shot (at 25%, 50% and 75% of the shot). These are used later for comparing how shots look.

**4. Transcript**
Parakeet TDT 0.6B v3 (through parakeet-mlx) converts the audio to text with timestamps for each sentence.
Backup options: Whisper large-v3-turbo, then the YouTube subtitles.

**5. Faces**
- Take 1 frame every second.
- InsightFace (buffalo_l) finds the faces (SCRFD detector) and turns each face into a 512 number "fingerprint" (ArcFace).
- Clear, front-facing faces are used to decide who is who. Blurry or side faces are kept but only tag along.
- Each face is cropped and saved.
- Faces of the same person inside one shot are linked together (tracks).
- The tracks are then clustered into people. Two faces in the same frame can never be the same person.
- If one person got split into two groups (different lighting etc.), the two groups get merged.
- Groups are named group1, group2... by screen time, so group1 is the person on screen the most.

**6. Finding scene cuts**
At every shot cut, check how much things change before vs after:
- how it looks: SigLIP2 on the keyframes (same room / same camera angles?)
- what they talk about: bge-small on the transcript (same topic?)
- who is there: the face groups (same people?)

Big change on these = likely a new scene. The strongest 60 cut points are kept as candidates.

**7. Gemini picks the scenes**
Gemini (gemini-3.8-flash, free tier) watches the full video straight from the YouTube link and picks which of those candidates actually start a new scene.
After that, small fix: if a scene starts with an outside/city shot with no faces, that shot goes with the scene it introduces.

**8. Scene descriptions**
The scenes are sent to Gemini in ~10 minute chunks. For each chunk it gets:
- the video clip
- a small photo strip of each face group
- the transcript of each scene

It returns the main theme, who is involved and what happens, and the description is built from that.
If one person is named in any scene (e.g. "Liza"), that name is used for their group everywhere.
If the Flash model runs out of free quota, it switches to gemini-3.5-flash-lite.

**9. Output**
- `output/output.json`: shots and scenes in the required format
- `output/faces/scene_XXX/groupN/`: face crops, file name = time in seconds
- `output/faces_tree.txt`: folder tree
- `output/output_extended.json`: extra details (scene titles, names, hh:mm:ss times)
- `output/qa/report.html`: quick visual check of every scene

**10. Checks**
A validation runs automatically at the end. For a separate full check:

```
uv run videoparse verify
```

It re-checks the JSON, the timings, that every timecode has its image, and that every face crop really belongs to its group.


## Models used

| Model | What it does |
|---|---|
| TransNetV2 | finds the shot cuts |
| Parakeet TDT 0.6B v3 | speech to text with timestamps |
| Whisper large-v3-turbo | backup speech to text |
| InsightFace buffalo_l (SCRFD + ArcFace) | finds faces and makes face embeddings for grouping |
| SigLIP2 (ViT-B-16) | compares how shots look |
| bge-small-en-v1.5 | compares what is being talked about |
| Gemini 3.8 Flash | picks the final scene cuts and writes the descriptions |
| Gemini 3.5 Flash-Lite | backup if Flash hits the free limit |

Everything runs on my Mac except Gemini, which is on the free API. Total cost: $0.
