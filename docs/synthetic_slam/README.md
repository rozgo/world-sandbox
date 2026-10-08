# Synthetic SLAM: run guide

A generated video goes through [AMB3R-SLAM](https://github.com/HengyiWang/amb3r-slam) (pinned at
`5418465` in `third_party/amb3r-slam`, unmodified) with a tracer that keeps every stage's results.
The journal is `site/journals/synthetic_slam/`; the brief is [BRIEF.md](BRIEF.md).

## 1. Generate the input (any machine, paid)

```sh
# WAVESPEED_API_KEY in the ignored .env. One submission per job folder; collect never resubmits.
uv run --locked python scripts/generate_video.py submit --prompt docs/synthetic_slam/prompts/junkyard_v1.txt \
    --job outputs/synthetic_slam/junkyard_v1 --duration 6
uv run --locked python scripts/generate_video.py collect --job outputs/synthetic_slam/junkyard_v1 --wait 900
```

`junkyard_v1`: Seedance 2.5 text-to-video, 720p, 16:9, 6 s, no audio; 1280 × 720 at 24 fps, 145 frames.
Record: [results/junkyard_v1_generation.json](results/junkyard_v1_generation.json); the video is kept
as `previews/synthetic_slam/junkyard_v1/input.mp4` (SHA-256 in the record).

## 2. Trace a SLAM run (Linux, NVIDIA GPU)

```sh
sudo apt install libopencv-dev                     # once; needed by the DBoW2 build
CUDA_HOME=/usr/local/cuda-13.0 scripts/setup_slam.sh  # env, native builds, weights (~6.5 GB)
uv run --locked --extra slam slam-trace --video previews/synthetic_slam/junkyard_v1/input.mp4 --fps 24 \
    --out outputs/synthetic_slam/junkyard_v1/trace
```

Tested on Ubuntu 24.04, RTX 4090, CUDA toolkit 13.0, torch 2.14.1+cu130, Python 3.11. The run used
the upstream defaults: monocular, uncalibrated, DA3-Nested-Giant-Large-1.1 with bf16 encoder weights,
DA3-Small front-end, DBoW2 retrieval, `--save_every 1` for the map.

The trace folder holds `trace.json` (configuration, timings, per-frame front-end records, every model
pass, submaps, alignments, long-context windows and covisibility, loop probes, pose graph),
`arrays.npz` (input images, poses, and each submap pass's depth, confidence, sky and intrinsics),
`map.npz` (points with source frame, submap and relative confidence) and `map.ply`.

## 3. Figures, film and viewer data (macOS or Linux)

```sh
uv run --locked slam-figures --trace outputs/synthetic_slam/junkyard_v1/trace \
    --video previews/synthetic_slam/junkyard_v1/input.mp4 --out previews/synthetic_slam/junkyard_v1
```

Writes the stage figures, `film.mp4` (1920 × 1080; the stored copy is re-encoded to 1600 px, CRF 24), `summary.json` and `viewer/scene.json` + `viewer/points.bin`
(1,000,000 points, three.js axes). `--only <step> ...` re-renders single items.

## 4. Journal

```sh
uv run --locked python scripts/build_journals.py
python3 -m http.server 8792 --bind 127.0.0.1 --directory build/journals
scripts/publish_pages.sh build/journals/synthetic_slam synthetic_slam
scripts/publish_pages.sh build/journals/hub .
```

## Recorded results

| File | What |
| --- | --- |
| [results/junkyard_v1_summary.json](results/junkyard_v1_summary.json) | the traced run: timings per stage, counts, submaps, alignments, long context, loops, intrinsics |
| [results/junkyard_v1_generation.json](results/junkyard_v1_generation.json) | the video generation request and record |
| [results/spires_christ_church_05.json](results/spires_christ_church_05.json) | an earlier reference run of the upstream code on a real 13-minute Oxford Spires sequence |

Headline numbers (junkyard_v1): 145 frames in 19.85 s (7.3 fps) after 10.9 s of model loading; 7.75 GB
peak GPU memory; 4 submaps from 6 passes; 5 alignments with relative residuals 0.41–1.61%; one
long-context link rejected (covisibility 1.4% < 20%); 29 loop probes, no matches; 3,330,749 map points.
The front-end's live scale estimate drifts from 1.0 to 140.6 by frame 96; the final trajectory comes
from the submaps.
