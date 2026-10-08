"""Run AMB3R-SLAM on a folder of frames and keep what every stage produced.

The upstream code in third_party/amb3r-slam is not modified. It runs as its own `run.py` runs
it; this module wraps methods on the live pipeline objects to record each stage:

  1 frames       the images at the model's input size
  2 front-end    a pose for every frame from DA3-Small, the keyframe it was tracked against
  3 submaps      each DA3 pass over a submap's frames: poses, depth, confidence, sky, intrinsics
  4 alignment    the Sim(3) joining each submap to the ones it overlaps, with residuals
  5 long context windows over many submaps, their covisibility checks and edges
  6 loops        retrieval probes, candidates and their verification
  7 pose graph   the submap poses before and after optimisation
  8 map          the fused points, each with the frame it came from

Runs on Linux with an NVIDIA GPU (`uv sync --extra slam`, then scripts/setup_slam.sh):

    uv run --extra slam slam-trace --video outputs/synthetic_slam/junkyard_v1/video.mp4 --fps 24 \
        --out outputs/synthetic_slam/junkyard_v1/trace
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
AMB3R = ROOT/"third_party"/"amb3r-slam"


def _f(x):
    """Tensors, arrays and numbers to plain JSON values."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    return x


class _NetProxy:
    """Stands in for a SLAMModel: times every forward pass and keeps its intrinsics."""

    def __init__(self, net, tracer, who):
        self._net, self._tracer, self._who = net, tracer, who

    def __getattr__(self, name):
        return getattr(self._net, name)

    def __call__(self, images, ref, calib_K=None):
        t0 = self._tracer.clock()
        out = self._net(images, ref, calib_K)
        self._tracer.on_pass(self._who, int(images.shape[0]), str(ref), self._tracer.clock() - t0, out)
        return out


class _DbProxy:
    """Stands in for the DBoW2 database: keeps each query's best score and match."""

    def __init__(self, db, tracer):
        self._db, self._tracer = db, tracer

    def __getattr__(self, name):
        return getattr(self._db, name)

    def insert_image(self, img):
        return self._db.insert_image(img)

    def query(self, n):
        score, j, matches = self._db.query(n)
        self._tracer.last_query = {"score": float(score), "match": int(j)}
        return score, j, matches


class Tracer:
    def __init__(self, pipeline):
        import torch
        self.torch = torch
        self.p, self.b = pipeline, pipeline.backend
        self.ctx = []
        self.passes, self.frontend, self.reanchors = [], [], []
        self.submaps, self.align, self.windows, self.covis = [], [], [], []
        self.probes, self.loop_checks, self.pgo = [], [], []
        self.seconds = defaultdict(float)
        self.last_query = None
        self.fe = None
        self.at = -1                # the frame being processed
        self._wrap()

    def clock(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        return time.perf_counter()

    def timed(self, stage, fn, *args, **kw):
        t0 = self.clock()
        try:
            return fn(*args, **kw)
        finally:
            self.seconds[stage] += self.clock() - t0

    # ---- hooks --------------------------------------------------------------------------
    def _wrap(self):
        b, t = self.b, self
        b.net = _NetProxy(b.net, self, "backend")

        frontend = self.p.frontend
        def make_frontend():
            fe = frontend()
            fe.net = _NetProxy(fe.net, t, "frontend")
            t._wrap_frontend(fe)
            t.fe = fe
            return fe
        self.p.frontend = make_frontend

        forward = b._forward
        def _forward(images, ref=None, frames=None, dense=False):
            t.ctx.append({"frames": None if frames is None else [int(f) for f in frames]})
            try:
                return forward(images, ref, frames, dense)
            finally:
                t.ctx.pop()
        b._forward = _forward

        new_submap = b._new_submap
        def _new_submap(a, bb, own):
            t.ctx.append({"stage": "submap"})
            try:
                t0 = t.clock()
                sm = new_submap(a, bb, own)
                t.seconds["3 submaps"] += t.clock() - t0
            finally:
                t.ctx.pop()
            t.on_submap(sm, a, bb, own)
            return sm
        b._new_submap = _new_submap

        record = b._record
        def _record(sm, replaces=None):
            if replaces is not None:
                for s in t.submaps:
                    if s["obj"] is replaces:
                        s["replaced_by"] = id(sm)
            return record(sm, replaces)
        b._record = _record

        align_pair = b._align_pair
        def _align_pair(a, bb):
            t0 = t.clock()
            g, info = align_pair(a, bb)
            t.seconds["4 alignment"] += t.clock() - t0
            t.align.append({"a": id(a), "b": id(bb), "shared": len(a.shared(bb)),
                            "g": None if g is None else _f(g), "info": _f(info)})
            return g, info
        b._align_pair = _align_pair

        lc_begin = b._long_context_begin
        def _long_context_begin():
            lc = lc_begin()
            cvm = lc.get("cvm")
            if cvm is not None:
                span_pair = cvm.span_pair
                def traced_span_pair(images, fa, fb, strong_at=0.05):
                    r = span_pair(images, fa, fb, strong_at=strong_at)
                    t.covis.append({"frames_a": [int(f) for f in fa], "frames_b": [int(f) for f in fb],
                                    **_f({k: v for k, v in r.items() if k != "best"}),
                                    "best": None if r.get("best") is None else list(r["best"])})
                    return r
                cvm.span_pair = traced_span_pair
            return lc
        b._long_context_begin = _long_context_begin

        window = b._long_context_window
        def _long_context_window(images, grp):
            t.ctx.append({"stage": "long_context"})
            n_pass = len(t.passes)
            try:
                edges = t.timed("5 long context", window, images, grp)
            finally:
                t.ctx.pop()
            t.windows.append({"submaps": [int(g) for g in grp],
                              "frames": next((p["frames"] for p in t.passes[n_pass:]), None),
                              "edges": [[int(e[0]), int(e[1])] for e in edges]})
            return edges
        b._long_context_window = _long_context_window

        loop_begin = b._loop_begin
        def _loop_begin():
            det = loop_begin()
            if det is not None and hasattr(det, "db"):
                det.db = _DbProxy(det.db, t)
            return det
        b._loop_begin = _loop_begin

        loop_query = b._loop_query
        def _loop_query(det, imgs, f):
            n = len(b._st.loop["pending"])
            t.last_query = None
            t0 = t.clock()
            loop_query(det, imgs, f)
            t.seconds["6 loops"] += t.clock() - t0
            if t.last_query is not None:
                pend = b._st.loop["pending"]
                t.probes.append({"frame": int(f), **t.last_query,
                                 "candidate": [int(x) for x in pend[-1][:2]] if len(pend) > n else None})
        b._loop_query = _loop_query

        verify = b._verify_loop
        def _verify_loop(imgs, fj, fi):
            before = dict(b.stats.get("loop_reject_reasons", {}))
            t.ctx.append({"stage": "loop"})
            try:
                edge = t.timed("6 loops", verify, imgs, fj, fi)
            finally:
                t.ctx.pop()
            after = b.stats.get("loop_reject_reasons", {})
            reason = next((k for k in after if after[k] != before.get(k, 0)), None)
            t.loop_checks.append({"frame": int(fj), "match": int(fi), "accepted": edge is not None,
                                  "submaps": None if edge is None else [int(edge[0]), int(edge[1])],
                                  "reason": reason})
            return edge
        b._verify_loop = _verify_loop

        optimise = b._optimise
        def _optimise(init):
            t0 = t.clock()
            out = optimise(init)
            t.seconds["7 pose graph"] += t.clock() - t0
            t.pgo.append({"before": _f(t.torch.stack(list(init))), "after": _f(out)})
            return out
        b._optimise = _optimise

    def _wrap_frontend(self, fe):
        t = self
        track = fe.track
        def traced_track(imgs, f):
            n_ctx = max(0, int(fe.cfg.frontend.context))
            kf = fe.kf_idx
            views = None if kf is None else sorted(dict.fromkeys(
                [kf] + [i for i, _ in fe.recent[-n_ctx:]] + [f]))
            t.at = int(f)
            t.ctx.append({"stage": "frontend", "frames": views})
            t.last_overlap = None
            try:
                pose = t.timed("2 front-end", track, imgs, f)
            finally:
                t.ctx.pop()
            t.frontend.append({"frame": int(f), "keyframe": None if kf is None else int(kf),
                               "views": views, "promoted": fe.kf_idx == f and kf != f,
                               "overlap": t.last_overlap, "conf": float(fe.last_conf),
                               "scale": None if fe.scale is None else float(fe.scale),
                               "pose": None if pose is None else _f(pose)})
            return pose
        fe.track = traced_track

        overlap = fe._overlap
        def traced_overlap(*a, **kw):
            t.last_overlap = float(overlap(*a, **kw))
            return t.last_overlap
        fe._overlap = traced_overlap

        reanchor = fe.reanchor
        def traced_reanchor(frames, P):
            t.reanchors.append({"after_frame": t.frontend[-1]["frame"] if t.frontend else None,
                                "frames": [int(f) for f in frames]})
            return reanchor(frames, P)
        fe.reanchor = traced_reanchor

    # ---- recorders ----------------------------------------------------------------------
    def on_pass(self, who, n_views, ref, seconds, out):
        _poses, _pts, _conf, K, scale = out
        ctx = {k: v for c in self.ctx for k, v in c.items()}
        self.passes.append({"model": who, "stage": ctx.get("stage", "?"), "frames": ctx.get("frames"), "at": self.at,
                            "views": n_views, "ref": ref, "seconds": seconds, "scale": float(scale)})
        self._last_K = None if K is None else K.detach().float().cpu().numpy()

    def on_submap(self, sm, a, b, own):
        pts, conf, sky = sm.dense if sm.dense is not None else (None, None, None)
        # Holding the object keeps its id unique: a freed warm-up submap's id can be reused.
        rec = {"id": id(sm), "obj": sm, "start": int(a), "end": int(b), "frames": [int(f) for f in own],
               "metric_scale": None if sm.metric_scale is None else float(sm.metric_scale),
               "seconds": self.passes[-1]["seconds"], "poses": sm.poses.detach().float().cpu().numpy(),
               "K": self._last_K}
        if pts is not None:
            T = sm.poses.float().to(pts.device)
            depth = ((pts.float() - T[:, None, None, :3, 3]) @ T[:, None, :3, :3])[..., 2]
            rec["depth"] = depth.half().cpu().numpy()
            rec["conf"] = conf.half().cpu().numpy()
            rec["sky"] = None if sky is None else sky.bool().cpu().numpy()
        self.submaps.append(rec)


def _recorder_class():
    from amb3r_slam.tools.export import MapRecorder, depth_edge, write_ply
    from amb3r_slam.tools.geometry import quat_to_rmat
    import torch

    class TracingMapRecorder(MapRecorder):
        """Upstream MapRecorder's selection and filtering, also remembering each point's frame."""

        def add(self, sm, images):
            pts, conf, sky = sm.dense
            mid = 0.5 * (sm.start + sm.end - 1)
            for i, f in enumerate(sm.frames):
                slot, dist = f // self.every, abs(f - mid)
                if slot in self.best and self.best[slot][0] <= dist:
                    continue
                p, c = pts[i].float(), conf[i].float()
                T = sm.poses[i].float().to(p.device)
                depth = ((p - T[:3, 3]) @ T[:3, :3])[..., 2]
                keep = (c >= torch.quantile(c.flatten(), self.conf_drop)) & (depth > 0)
                keep &= ~depth_edge(depth, self.edge_rtol)
                if sky is not None:
                    keep &= ~sky[i].to(keep.device)
                col = ((images[f].permute(1, 2, 0).to(keep.device) + 1.0) * 127.5).clamp(0, 255)
                grid = torch.zeros_like(keep)
                grid[::self.stride, ::self.stride] = True
                self.best[slot] = (dist, sm, p[keep].half().cpu(), col[keep].byte().cpu(),
                                   grid[keep].cpu(), int(f), (c[keep] / c.median()).half().cpu())

        def drop(self, sm):
            self.best = {s: v for s, v in self.best.items() if v[1] is not sm}

        def save(self, path, result):
            index = {id(sm): k for k, sm in enumerate(result.submaps)}
            xyz, rgb, on, frame, sub, rel = [], [], [], [], [], []
            for slot in sorted(self.best):
                _, sm, p, col, g_on, f, c = self.best[slot]
                k = index.get(id(sm))
                if k is None:
                    continue
                g = result.submap_poses[k].double().cpu()
                R, t, s = quat_to_rmat(g[3:7]), g[:3], g[7]
                xyz.append((s * p.double() @ R.T + t).float().numpy())
                rgb.append(col.numpy())
                on.append(g_on.numpy())
                frame.append(np.full(len(p), f, np.int16))
                sub.append(np.full(len(p), k, np.int16))
                rel.append(c.numpy())
            xyz, rgb, on, frame, sub, rel = (np.concatenate(x) for x in (xyz, rgb, on, frame, sub, rel))
            write_ply(path, xyz, rgb)
            write_ply(path[:-len(".ply")] + "_stride.ply", xyz[on], rgb[on])
            self.cloud = {"xyz": xyz, "rgb": rgb, "frame": frame, "submap": sub, "conf_rel": rel}
            return len(xyz), int(on.sum())

    return TracingMapRecorder


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="a video; its frames are extracted losslessly to <out>/frames")
    src.add_argument("--frames", help="a folder of frames, in name order")
    ap.add_argument("--fps", type=float, required=True, help="frame rate of the folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_every", type=int, default=1, help="keep one of every n mapped frames")
    ap.add_argument("--pixel_stride", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.video:
        import subprocess
        import imageio_ffmpeg
        a.frames = str(out/"frames")
        os.makedirs(a.frames, exist_ok=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(Path(a.video).resolve()),
                        str(Path(a.frames)/"%05d.png")], check=True)

    sys.path.insert(0, str(AMB3R))
    os.chdir(AMB3R)
    import torch
    import run as upstream  # third_party/amb3r-slam/run.py: its argument parser and torch settings
    from omegaconf import OmegaConf
    from amb3r_slam import datasets
    from amb3r_slam.model import load_model
    from amb3r_slam.pipeline import AMB3R_SLAM

    sys.argv = ["run.py", "--dataset", "demo", "--data_path", str(Path(a.frames).resolve()),
                "--fps", str(a.fps), "--out", str(out)]
    args = upstream.arguments()
    ds = datasets.get(args.dataset)(args, args.model_name)
    resolution = ds.resolution()
    t0 = time.perf_counter()
    model = load_model(args.model_name, args.ckpt_path, args.device, fp32=args.fp32).eval()
    pipeline = AMB3R_SLAM(model, cfg_path=args.config, modality=args.sensor, overrides=None,
                          fps=args.fps or ds.rate(), device=args.device)
    load_s = time.perf_counter() - t0
    seq = next(iter(ds.sequences(resolution)))
    n = seq.images.shape[1]
    tracer = Tracer(pipeline)
    pipeline.reset(calib_K=None)
    ds.attach(seq, pipeline.backend, pipeline.cfg, args.sensor, resolution)
    pipeline.backend.recorder = _recorder_class()(pipeline.cfg.backend.frame_stride, a.save_every, a.pixel_stride)
    torch.cuda.reset_peak_memory_stats()
    print(f"{seq.name}: {n} frames at {resolution[0]}x{resolution[1]}", flush=True)

    t0 = tracer.clock()
    mem = pipeline.run(seq.images, verbose=True)
    run_s = tracer.clock() - t0
    t1 = time.perf_counter()
    counts = pipeline.backend.recorder.save(str(out/"map.ply"), mem)
    tracer.seconds["8 map"] += time.perf_counter() - t1
    peak = torch.cuda.max_memory_allocated() / 1e9

    # Submaps: committed ones get their final index; replaced warm-up submaps are kept, marked.
    final = {id(sm): k for k, sm in enumerate(mem.submaps)}
    ids = {s["id"]: i for i, s in enumerate(tracer.submaps)}
    st = pipeline.backend._st
    frames = seq.images[0]
    images = np.stack([((frames[i].float().permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).byte().numpy()
                       for i in range(n)])
    arrays = {"images": images, "poses": mem.poses[:n].numpy(), "online": st.online.numpy(),
              "keyframes": np.array(mem.keyframes),
              "submap_poses": torch.stack(list(mem.submap_poses)).float().cpu().numpy(),
              "S_odo": torch.stack(list(st.S_odo)).float().cpu().numpy()}
    track = np.full((n, 4, 4), np.nan, np.float32)
    for r in tracer.frontend:
        if r["pose"] is not None:
            track[r["frame"]] = np.asarray(r["pose"], np.float32)
    arrays["frontend_track"] = track
    submaps = []
    for i, s in enumerate(tracer.submaps):
        k = None if "replaced_by" in s else final.get(s["id"])
        for key in ("poses", "K", "depth", "conf", "sky"):
            if s.get(key) is not None:
                arrays[f"pass{i}_{key}"] = s[key]
        submaps.append({"pass": i, "index": k, "start": s["start"], "end": s["end"], "frames": s["frames"],
                        "metric_scale": s["metric_scale"], "seconds": s["seconds"],
                        "replaced": "replaced_by" in s})
    index = lambda sid: final.get(sid)  # noqa: E731
    align = [{**e, "a": index(e["a"]), "b": index(e["b"]),
              "a_pass": ids.get(e["a"]), "b_pass": ids.get(e["b"])} for e in tracer.align]
    np.savez_compressed(out/"arrays.npz", **arrays)
    np.savez_compressed(out/"map.npz", **pipeline.backend.recorder.cloud)

    trace = {
        "frames": {"folder": str(Path(a.frames).resolve().name), "count": n, "fps": a.fps,
                   "resolution": list(resolution)},
        "model": {"backend": pipeline.cfg.get("model_ckpt", args.model_name), "frontend": str(pipeline.cfg.frontend.ckpt),
                  "device": torch.cuda.get_device_name(0)},
        "config": OmegaConf.to_container(pipeline.cfg, resolve=True),
        "timing": {"load_s": load_s, "run_s": run_s, "fps": n / run_s,
                   "stages_s": dict(sorted(tracer.seconds.items())),
                   "model_passes_s": {w: sum(p["seconds"] for p in tracer.passes if p["model"] == w)
                                      for w in ("frontend", "backend")}},
        "peak_gpu_gb": peak,
        "counts": {"submaps": len(mem.submaps), "submap_passes": len(tracer.submaps),
                   "frontend_passes": sum(p["model"] == "frontend" for p in tracer.passes),
                   "backend_passes": sum(p["model"] == "backend" for p in tracer.passes),
                   "keyframes_promoted": sum(r["promoted"] for r in tracer.frontend),
                   "align_edges": len(align), "span_edges": len(st.span),
                   "long_context_windows": len(tracer.windows), "long_context_edges": len(st.long_context),
                   "probes": len(tracer.probes), "loop_candidates": len(tracer.loop_checks), "loops": len(st.loops),
                   "map_points": counts[0], "map_points_stride": counts[1]},
        "frontend": [{k: v for k, v in r.items() if k != "pose"} for r in tracer.frontend],
        "reanchors": tracer.reanchors,
        "passes": tracer.passes,
        "submaps": submaps,
        "align": align,
        "span": [[int(e[0]), int(e[1]), _f(e[2]), _f(e[3])] for e in st.span],
        "windows": tracer.windows,
        "covis": tracer.covis,
        "long_context": [[int(e[0]), int(e[1]), _f(e[2])] for e in st.long_context],
        "probes": tracer.probes,
        "loop_checks": tracer.loop_checks,
        "loops": [[int(e[0]), int(e[1]), _f(e[2])] for e in st.loops],
        "pgo": tracer.pgo,
        "stats": _f(mem.stats),
    }
    (out/"trace.json").write_text(json.dumps(trace, indent=1))
    print(json.dumps({"run_s": round(run_s, 2), "fps": round(n / run_s, 2), "peak_gpu_gb": round(peak, 2),
                      **trace["counts"]}), flush=True)


if __name__ == "__main__":
    main()
