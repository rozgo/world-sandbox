# Synthetic SLAM: run guide

A generated video goes through [AMB3R-SLAM](https://github.com/HengyiWang/amb3r-slam) with a tracer that
keeps every stage's results. `third_party/amb3r-slam` points at our fork
[rozgo/amb3r-slam](https://github.com/rozgo/amb3r-slam/tree/fix/reanchor-units), branch `fix/reanchor-units`,
commit `552e17f`: the authors' `5418465` plus two fixes to the front-end (see "The front-end fix" below).
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
uv run --locked slam-figures --trace outputs/synthetic_slam/junkyard_v1/trace_fix \
    --video previews/synthetic_slam/junkyard_v1/input.mp4 --out previews/synthetic_slam/junkyard_v1_fix
# the run before the fix against the run after it: numbers and stage2_handoff_fix.png
uv run --locked python -m world_sandbox.slam.compare --before outputs/synthetic_slam/junkyard_v1/trace \
    --after outputs/synthetic_slam/junkyard_v1/trace_fix \
    --json docs/synthetic_slam/results/junkyard_v1_reanchor_fix.json --fig previews/synthetic_slam/junkyard_v1_fix
```

`previews/synthetic_slam/junkyard_v1/` keeps the figures of the first run, before the fix
(`outputs/synthetic_slam/junkyard_v1/trace`, upstream `5418465`); the journal uses `junkyard_v1_fix/`.

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
| [results/junkyard_v1_fix_summary.json](results/junkyard_v1_fix_summary.json) | the traced run the journal shows (fork `552e17f`): timings per stage, counts, submaps, alignments, long context, loops, intrinsics |
| [results/junkyard_v1_summary.json](results/junkyard_v1_summary.json) | the first traced run, before the fix (upstream `5418465`) |
| [results/junkyard_v1_reanchor_fix.json](results/junkyard_v1_reanchor_fix.json) | the front-end's live estimate against the final path, before and after the fix |
| [results/spires_christ_church_05_live.json](results/spires_christ_church_05_live.json) | the live estimate and final path against ground truth on the Oxford walk: upstream, first fix, both fixes |
| [results/junkyard_v1_generation.json](results/junkyard_v1_generation.json) | the video generation request and record |
| [results/spires_christ_church_05.json](results/spires_christ_church_05.json) | an earlier reference run of the upstream code on a real 13-minute Oxford Spires sequence |

Headline numbers (junkyard_v1, fork `552e17f`): 145 frames in 20.62 s (7.0 fps) after 11.5 s of model loading;
7.8 GB peak GPU memory; 4 submaps from 6 passes; 5 alignments with relative residuals 0.54–1.51%; one
long-context link rejected (covisibility 1.4% < 20%); 29 loop probes, no matches; 3,330,429 map points.

## The front-end fix

The first run (upstream `5418465`) showed the front-end's live estimate running away from the final path:
21.5 times too long. Two causes. At each hand-off the backend overwrote the poses of its submap's frames (about
one in four) with its own, in another origin and scale, so the front-end's next fits mixed two worlds. And the
front-end took each pass's scale from the distances to frames it had just placed itself, so errors compounded.
The fork's two commits: `692067a` carries every front-end pose into the backend's world at each hand-off and
aligns each warm-up submap to the one it replaces; `552e17f` takes the scale from the keyframe's depth and
restarts the front-end from the backend's newest frames at each hand-off. Here the live path is now 1.05 times
the final path's length, within 0.9% of it (RMS, after a Sim(3) fit).

On the Oxford walk (Linux + GPU; `run.py`'s settings, plus the pose the front-end returned for every frame):

```sh
uv run --locked --extra slam python -m world_sandbox.slam.live_eval --dataset spires \
    --data_path <spires root> --only 2024-03-20-christ-church-05 --verbose --out <dir>
```

the live estimate's error against ground truth fell from 39.7 m to 3.9 m (ATE over 813 m). Still to do: the
live pose snaps back at hand-offs (up to 3.2 m there), and the final path moved from 1.85 m to 2.29 m on this
one sequence; more sequences will tell whether that is a cost of the fix.
