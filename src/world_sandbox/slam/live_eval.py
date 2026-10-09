"""The front-end's live estimate, and the final trajectory, against a dataset's ground truth.

Upstream `run.py`'s arguments and setup, plus a record of the pose the front-end returns for
every frame: the estimate a robot would act on while it moves. Linux with an NVIDIA GPU.

    uv run --locked --extra slam python -m world_sandbox.slam.live_eval --dataset spires \
        --data_path <spires root> --only 2024-03-20-christ-church-05 --out <dir>

Writes, per sequence, `live_<seq>.npz` (final, live, online and ground-truth poses, keyframes)
and `live_<seq>.json`: absolute trajectory error (Sim(3)-aligned, as upstream's evaluation),
a short-range relative error and the live-to-final agreement, with the amb3r-slam commit.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AMB3R = Path(__file__).resolve().parents[3]/"third_party"/"amb3r-slam"


def umeyama(src, dst):
    """Sim(3) (s, R, t) minimising |dst - (s R src + t)| over corresponding points."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


def score(est, gt, delta):
    """Positions of ``est`` Sim(3)-aligned onto ``gt``: ATE (RMS), and the error of each
    ``delta``-frame displacement (RMS and 95th percentile), in ground-truth metres."""
    s, R, t = umeyama(est, gt)
    a = (s * (R @ est.T)).T + t
    ate = float(np.sqrt((np.linalg.norm(a - gt, axis=1) ** 2).mean()))
    d = np.linalg.norm((a[delta:] - a[:-delta]) - (gt[delta:] - gt[:-delta]), axis=1)
    return {"ate_m": round(ate, 4), "sim3_scale": round(float(s), 4),
            f"rpe_{delta}f_rms_m": round(float(np.sqrt((d ** 2).mean())), 4),
            f"rpe_{delta}f_p95_m": round(float(np.percentile(d, 95)), 4),
            f"rpe_{delta}f_max_m": round(float(d.max()), 4)}


def main():
    # run.py's arguments; --out is resolved here, before moving into the checkout.
    for i, a in enumerate(sys.argv[:-1]):
        if a == "--out":
            sys.argv[i + 1] = str(Path(sys.argv[i + 1]).resolve())
    sys.path.insert(0, str(AMB3R))
    os.chdir(AMB3R)
    import torch
    import run as upstream  # third_party/amb3r-slam/run.py: its arguments and torch settings
    from amb3r_slam import datasets
    from amb3r_slam.model import load_model
    from amb3r_slam.pipeline import AMB3R_SLAM

    args = upstream.arguments()
    if not args.out:
        raise SystemExit("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *c: subprocess.run(["git", "-C", str(AMB3R), *c], capture_output=True, text=True).stdout.strip()  # noqa: E731
    commit = {"commit": git("rev-parse", "HEAD"),
              "modified": bool(git("status", "--porcelain", "--untracked-files=no"))}

    ds = datasets.get(args.dataset)(args, args.model_name)
    resolution = ds.resolution()
    model = load_model(args.model_name, args.ckpt_path, args.device, fp32=args.fp32).eval()
    pipeline = AMB3R_SLAM(model, cfg_path=args.config, modality=args.sensor,
                          overrides=upstream.parse_overrides(args.override),
                          fps=args.fps or ds.rate(), device=args.device)

    live = {}
    make_frontend = pipeline.frontend
    def frontend():
        fe = make_frontend()
        track = fe.track
        def traced(imgs, f):
            pose = track(imgs, f)
            if pose is not None:
                live[int(f)] = pose.detach().float().cpu().numpy()
            return pose
        fe.track = traced
        return fe
    pipeline.frontend = frontend

    for seq in ds.sequences(resolution):
        if not ds.wanted(seq.name):
            continue
        n = seq.images.shape[1]
        name = seq.name.replace("/", "_")
        print(f"=== {seq.name}: {n} frames, amb3r-slam {commit['commit'][:7]} ===", flush=True)
        live.clear()
        torch.cuda.reset_peak_memory_stats()
        pipeline.reset(calib_K=None)
        ds.attach(seq, pipeline.backend, pipeline.cfg, args.sensor, resolution)
        t0 = time.time()
        mem = pipeline.run(seq.images, verbose=args.verbose)
        dt = time.time() - t0

        final = mem.poses[:n].numpy().astype(np.float64)
        track = np.full((n, 4, 4), np.nan)
        for f, p in live.items():
            track[f] = p
        online = mem.frontend_poses[:n].numpy() if mem.frontend_poses is not None else np.zeros((0, 4, 4))
        gt = np.asarray(seq.poses, dtype=np.float64)
        idx = np.asarray(seq.gt_index) if seq.gt_index is not None and len(seq.gt_index) else np.arange(n)
        ok = idx[np.isfinite(track[idx, 0, 3])]
        g = gt[: len(idx)] if len(gt) == len(idx) else gt
        gpos = {int(f): g[k][:3, 3] for k, f in enumerate(idx)}
        G = np.array([gpos[int(f)] for f in ok])
        fps_cam = ds.rate() if hasattr(ds, "rate") else None
        delta = 5
        res = {
            "sequence": seq.name, "frames": int(n), "amb3r_slam": commit,
            "time_s": round(dt, 1), "fps": round(n / dt, 2),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "loops": int(pipeline.stats.get("n_loops", 0)), "keyframes": len(mem.keyframes),
            "gt_path_m": round(float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum()), 1),
            "rpe_delta_frames": delta, "camera_rate_hz": fps_cam,
            "final": score(final[ok][:, :3, 3], G, delta),
            "live": score(track[ok][:, :3, 3], G, delta),
        }
        lf = track[ok][:, :3, 3]
        s, R, t = umeyama(lf, final[ok][:, :3, 3])
        res["live_vs_final_after_sim3_rms"] = round(float(np.sqrt((np.linalg.norm((s * (R @ lf.T)).T + t - final[ok][:, :3, 3], axis=1) ** 2).mean())), 4)
        np.savez_compressed(out/f"live_{name}.npz", final=final.astype(np.float32), live=track.astype(np.float32),
                            online=online, gt=gt, gt_index=idx, kf=np.array(mem.keyframes))
        (out/f"live_{name}.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
