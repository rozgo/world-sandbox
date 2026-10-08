# Synthetic SLAM: brief

Public-safe summary of the request (October 8, 2026).

- Start a World Sandbox repository organised like the MuJoCo Sandbox: experiments with journals,
  published to its own GitHub Pages site, with room for more projects.
- First journal: explain, very visually, how ordinary RGB video becomes a 3D reconstruction with SLAM,
  stage by stage, and what each stage does. Cover the input (one RGB camera, no calibration), the
  feed-forward model, the hierarchical backend, speed against real time, and when to choose this system.
- Break AMB3R-SLAM into explicit stages in our own code and capture each stage's intermediate results,
  so stages can be replaced or improved later. Pin AMB3R-SLAM as an unmodified git submodule.
- Input: a short generated video rather than a dataset: about 6 s of what a small robot would see,
  in an industrial junkyard with cars, generated with Seedance 2.5 text-to-video through WaveSpeed.
  The user approved the generated clip before SLAM ran on it. Loop closure is not a goal for this clip.
- A 3D viewer: by default a follow camera behind the computed SLAM camera, showing its pose and frustum;
  all recorded data as 3D overlays and debug views; an optional ego view through the camera.

## Done when

- The generated video, its prompt and its generation record are in the repository.
- A traced run produces a camera path and a map, and the trace holds every stage's output.
- The journal shows each stage with figures made from that trace, a film, and the embedded viewer.
- The journal and the sandbox home page are published on GitHub Pages.
