# Working in the World Sandbox

These instructions apply throughout this repository. Follow explicit user instructions when they differ.

## Start with the brief

- Save each project's brief in `docs/<project>/BRIEF.md`: the goal, the inputs, what must be shown and how
  it will be judged. Keep it public-safe: no machine names, connection details, credentials or private
  project names, in docs, logs, code comments or media.
- Check `git status` and preserve unrelated work. Use a separate package or module, CLI and output folder
  for each project; keep earlier versions of a result when making a new one (`junkyard_v1`, `_v2`, ...).

## Repository map

| Project | Code | CLI | Read first |
| --- | --- | --- | --- |
| Synthetic SLAM | `src/world_sandbox/slam/` | `slam-trace` (GPU), `slam-figures` | [run guide](docs/synthetic_slam/README.md) |

```sh
uv sync --locked                                  # macOS or Linux: figures, films, journals
scripts/setup_slam.sh                             # Linux + NVIDIA GPU: the SLAM runtime (extra "slam")
uv run --locked python scripts/build_journals.py  # build every journal and the home page
```

Use `uv`, `pyproject.toml` and `uv.lock`; keep heavy stacks in extras. Avoid global installs and
unrelated upgrades.

## Third-party code stays pinned

- Upstream projects live in `third_party/` as git submodules at a recorded commit. Observe or extend them
  from `src/` (the SLAM tracer wraps methods on live objects; its map recorder is a subclass). Native build
  products stay untracked inside the submodule.
- When upstream code itself is wrong, fix it in a fork under `rozgo/`, on a branch, and point the submodule
  at that commit (AMB3R-SLAM: `fix/reanchor-units`). Never edit a submodule without committing to its fork;
  say in the journal what changed and keep the run before the fix for comparison.
- Record each model's and dataset's licence in the journal footer.

## Generated inputs

- Generated video is an input, not a measurement. Label it as generated wherever it appears.
- Paid generation: `scripts/generate_video.py submit` creates exactly one task per job folder and saves
  its ID; `collect` only polls and downloads. Never resubmit to retry; resume the saved task. Keep the
  prompt file, the request and the output's SHA-256 under `docs/<project>/`. Keys live in the ignored
  `.env` only.

## GPU work

- Run GPU jobs on a machine you are authorised to use, from a checkout of this repository at a known
  commit, through `uv run --locked --extra slam`. Copy back the trace folder and keep its logs; record
  the GPU, CUDA and torch versions with the results. Keep connection details out of Git.
- Keep runs reproducible: same input video, same commit, same submodule commit, recorded config (the
  trace stores the resolved configuration).

## Results, figures and journals

- Every number in a journal comes from a file in the repository (`docs/<project>/results/`). State the
  conditions; keep failures and surprises on record (the front-end hand-off bug is an example).
- Figures, films and viewer data are built from saved traces by code in `src/`; regenerate them rather
  than editing images. Inspect every figure and sampled film frames before publishing.
- Journals follow [site/journals/README.md](site/journals/README.md). Build, serve locally, check the
  page and the viewer in a browser, commit the sources, then publish with `scripts/publish_pages.sh`.
