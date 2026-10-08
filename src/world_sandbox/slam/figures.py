"""Figures, the hero film and the 3D viewer's data, all from one `slam-trace` run.

    uv run slam-figures --trace outputs/synthetic_slam/junkyard_v1/trace \
        --video outputs/synthetic_slam/junkyard_v1/video.mp4 --out previews/synthetic_slam/junkyard_v1

Writes stage figures (PNG/JPEG), `film.mp4` (input and reconstruction side by side), and
`viewer/scene.json` + `viewer/points.bin` for the three.js viewer. Runs on the Mac (no GPU).
"""

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from world_sandbox.slam.render import (UP, View, chase_view, draw_frustum, draw_path, look_at,
                                       smooth_poses, splat)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
BG, PANEL, TEXT, MUTED, GRID = "#111315", "#181b1e", "#e8e4dd", "#9a9690", "#2a2e33"
RUST, TEAL, STEEL, YELLOW, RED, VIOLET = "#e0703a", "#4fb3a9", "#8d97a1", "#e8b931", "#e0524f", "#a48cf0"
SUBMAP = [TEAL, RUST, YELLOW, VIOLET, "#5b9bd5", "#7cc96b"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
                     "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "legend.facecolor": PANEL, "legend.edgecolor": GRID,
                     "font.family": "DejaVu Sans"})


def rgb(hexcol):
    return tuple(int(hexcol[i:i + 2], 16) for i in (1, 3, 5))


def font(size):
    for name in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Run:
    """One trace: the JSON record, its arrays and its map."""

    def __init__(self, trace_dir, video):
        d = Path(trace_dir)
        self.t = json.loads((d/"trace.json").read_text())
        self.a = np.load(d/"arrays.npz")
        self.m = np.load(d/"map.npz")
        self.video = Path(video)
        self.images = self.a["images"]
        self.n = len(self.images)
        self.H, self.W = self.images.shape[1:3]
        self.poses = self.a["poses"].astype(np.float64)
        self.fps = float(self.t["frames"]["fps"])
        self.committed = [s for s in self.t["submaps"] if s["index"] is not None]
        Ks = np.concatenate([self.a[f"pass{s['pass']}_K"] for s in self.committed])
        self.K = np.median(Ks, 0)
        self.path_len = float(np.linalg.norm(np.diff(self.poses[:, :3, 3], axis=0), axis=1).sum())

    def video_frames(self, width, height):
        """Every frame of the source video, resized: (T, H, W, 3) uint8."""
        raw = subprocess.run([FFMPEG, "-loglevel", "error", "-i", str(self.video), "-vf",
                              f"scale={width}:{height}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             check=True, capture_output=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)

    def submap_points(self, pass_i, conf_q=0.2, every=1):
        """A pass's points in its own frame, from depth, intrinsics and its poses."""
        depth = self.a[f"pass{pass_i}_depth"].astype(np.float32)
        conf = self.a[f"pass{pass_i}_conf"].astype(np.float32)
        sky = self.a[f"pass{pass_i}_sky"]
        K, P = self.a[f"pass{pass_i}_K"], self.a[f"pass{pass_i}_poses"]
        frames = self.t["submaps"][pass_i]["frames"]
        v, u = np.mgrid[0:self.H:every, 0:self.W:every]
        xyz, col, idx = [], [], []
        for i, f in enumerate(frames):
            d, c = depth[i, ::every, ::every], conf[i, ::every, ::every]
            keep = (c >= np.quantile(conf[i], conf_q)) & (d > 0) & ~sky[i, ::every, ::every]
            x = (u[keep] - K[i, 0, 2]) / K[i, 0, 0] * d[keep]
            y = (v[keep] - K[i, 1, 2]) / K[i, 1, 1] * d[keep]
            cam = np.stack([x, y, d[keep]], 1)
            xyz.append(cam @ P[i, :3, :3].T + P[i, :3, 3])
            col.append(self.images[f, ::every, ::every][keep])
            idx.append(np.full(keep.sum(), i))
        return np.concatenate(xyz), np.concatenate(col), np.concatenate(idx)


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  wrote", path.name)


def turbo_depth(d):
    """Inverse depth through the turbo colormap, robustly scaled; near is warm."""
    inv = 1.0 / np.maximum(d, 1e-6)
    lo, hi = np.quantile(inv[np.isfinite(inv)], [0.02, 0.98])
    return (plt.get_cmap("turbo")(np.clip((inv - lo) / (hi - lo + 1e-9), 0, 1))[..., :3] * 255).astype(np.uint8)


# ---- stage figures --------------------------------------------------------------------------
def fig_input(run, out):
    picks = np.linspace(0, run.n - 1, 8).round().astype(int)
    frames = run.video_frames(640, 360)
    fig, axes = plt.subplots(2, 4, figsize=(16, 4.9))
    for ax, f in zip(axes.flat, picks):
        ax.imshow(frames[f])
        ax.set_title(f"{f / run.fps:.1f} s · frame {f}", fontsize=11, color=MUTED, fontweight="normal")
        ax.axis("off")
    fig.subplots_adjust(wspace=0.04, hspace=0.18)
    save(fig, out/"input_filmstrip.png")


def fig_frames(run, out):
    f = run.n // 2
    full = run.video_frames(1280, 720)[f]
    small = run.images[f].copy()
    patch = 14
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1]})
    a.imshow(full)
    a.set_title(f"Generated frame {f}: 1280 × 720 px", loc="left")
    b.imshow(small, extent=(0, run.W, run.H, 0))
    for x in range(0, run.W + 1, patch):
        b.axvline(x, color="white", lw=0.35, alpha=0.45)
    for y in range(0, run.H + 1, patch):
        b.axhline(y, color="white", lw=0.35, alpha=0.45)
    b.add_patch(plt.Rectangle((patch * 18, patch * 10), patch, patch, fill=False, ec=RUST, lw=2))
    b.set_title(f"What the model sees: {run.W} × {run.H} px = {run.W // patch} × {run.H // patch} "
                f"patches of {patch} px", loc="left")
    for ax in (a, b):
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, out/"stage1_frames.png")


def fig_frontend(run, out):
    t = run.t
    fe = t["frontend"]
    ex = next(r for r in fe if r["frame"] == run.n * 2 // 5)
    views = ex["views"]
    cfg = t["config"]["frontend"]
    fig = plt.figure(figsize=(16, 9.2))
    gs = fig.add_gridspec(2, len(views), height_ratios=[1, 1.45], hspace=0.28, wspace=0.05)
    for i, v in enumerate(views):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(run.images[v])
        label = "keyframe" if v == ex["keyframe"] else ("new frame" if v == ex["frame"] else "recent")
        col = RUST if label == "new frame" else (TEAL if label == "keyframe" else STEEL)
        ax.set_title(f"{label} · frame {v}", color=col, fontsize=12)
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
    bottom = gs[1, :].subgridspec(1, 4, wspace=0.32)
    ax = fig.add_subplot(bottom[0, :2])
    fr = [r["frame"] for r in fe if r["overlap"] is not None]
    ov = [r["overlap"] for r in fe if r["overlap"] is not None]
    ax.plot(fr, ov, color=TEAL, lw=2, label="keyframe still visible in the new frame")
    ax.axhline(cfg["min_overlap"], color=RUST, ls="--", lw=1.5, label=f"promote below {cfg['min_overlap']}")
    for r in fe:
        if r["promoted"] and r["frame"] > 0:
            ax.axvline(r["frame"], color=YELLOW, lw=1, alpha=0.6)
    ax.plot([], [], color=YELLOW, lw=1, label="new keyframe")
    ax.set_xlim(0, run.n - 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("frame"); ax.set_ylabel("overlap")
    ax.set_title("When does the tracker pick a new keyframe?", loc="left")
    ax.grid(alpha=0.5); ax.legend(loc="lower left", fontsize=10)

    ax = fig.add_subplot(bottom[0, 2])
    sc = [(r["frame"], r["scale"]) for r in fe if r["scale"]]
    ax.semilogy(*zip(*sc), color=STEEL, lw=2)
    for r in t["reanchors"]:
        ax.axvline(r["after_frame"], color=TEAL, lw=1.2, alpha=0.8)
    ax.plot([], [], color=TEAL, lw=1.2, label="re-anchored by the backend")
    ax.set_xlim(0, run.n - 1); ax.set_xlabel("frame")
    ax.set_title("Front-end scale, live", loc="left"); ax.grid(alpha=0.5, which="both")
    ax.legend(fontsize=9, loc="upper left")

    ax = fig.add_subplot(bottom[0, 3])
    fin = run.poses[:, :3, 3]
    kf = [r["frame"] for r in fe if r["promoted"]]
    ax.plot(fin[:, 0], fin[:, 2], color=RUST, lw=2.5, label="final path")
    ax.scatter(fin[kf, 0], fin[kf, 2], s=40, color=YELLOW, zorder=5, label="keyframes")
    ax.scatter(*fin[0, [0, 2]], s=90, marker="o", facecolor="none", edgecolor=TEXT, lw=2, zorder=6, label="start")
    ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("z, forward")
    ax.set_title("Final path, from above", loc="left"); ax.legend(fontsize=9, loc="lower left")
    save(fig, out/"stage2_frontend.png")


def example_pass(run):
    """The committed submap pass in the middle of the sequence."""
    return run.committed[len(run.committed) // 2]["pass"]


def fig_submap(run, out):
    p = example_pass(run)
    s = run.t["submaps"][p]
    frames = s["frames"]
    picks = np.linspace(0, len(frames) - 1, 6).round().astype(int)
    depth, conf = run.a[f"pass{p}_depth"].astype(np.float32), run.a[f"pass{p}_conf"].astype(np.float32)
    sky = run.a[f"pass{p}_sky"]
    fig, axes = plt.subplots(4, 6, figsize=(16, 7.6))
    rows = ["input", "depth (near = warm)", "confidence", "sky mask"]
    for j, i in enumerate(picks):
        f = frames[i]
        axes[0, j].imshow(run.images[f]); axes[0, j].set_title(f"frame {f}", fontsize=11, color=MUTED)
        axes[1, j].imshow(turbo_depth(depth[i]))
        c = conf[i] / np.median(conf[i])
        axes[2, j].imshow(np.clip(c, 0, 3), cmap="viridis", vmin=0, vmax=3)
        sk = run.images[f].copy()
        sk[sky[i]] = (0.35 * sk[sky[i]] + 0.65 * np.array(rgb(TEAL))).astype(np.uint8)
        axes[3, j].imshow(sk)
    for r, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(rows[r], fontsize=11, color=TEXT)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(wspace=0.03, hspace=0.06)
    save(fig, out/"stage3_submap.png")


def fig_submap3d(run, out):
    p = example_pass(run)
    xyz, col, _ = run.submap_points(p, every=1)
    P = run.a[f"pass{p}_poses"].astype(np.float64)
    centre = np.median(xyz, 0)
    mid = P[len(P) // 2, :3, 3]
    span = np.linalg.norm(P[-1, :3, 3] - P[0, :3, 3])
    eye = mid - 1.2 * span * P[len(P) // 2, :3, 2] + 1.0 * span * UP + 0.6 * span * P[len(P) // 2, :3, 0]
    view = View(*look_at(eye, centre), size=(1600, 900))
    img = Image.fromarray(splat(view, xyz, col))
    for i, pose in enumerate(P):
        if i % 2 == 0:
            draw_frustum(img, view, pose, run.a[f"pass{p}_K"][i], (run.W, run.H), 0.18 * span / 3, rgb(TEAL), 2)
    draw_path(ImageDraw.Draw(img), view, P[:, :3, 3], rgb(RUST), 3)
    img.save(out/"stage3_submap3d.jpg", quality=90)
    print("  wrote stage3_submap3d.jpg")


def fig_spans(run, out):
    t = run.t
    fig, ax = plt.subplots(figsize=(16, 5.4))
    y = 0
    labels = []
    for s in t["submaps"]:
        col = SUBMAP[s["index"]] if s["index"] is not None else STEEL
        ax.add_patch(plt.Rectangle((s["start"], y - 0.32), s["end"] - s["start"], 0.64,
                                   color=col, alpha=0.18 if s["index"] is not None else 0.1,
                                   hatch=None if s["index"] is not None else "//", ec=col))
        ax.scatter(s["frames"], [y] * len(s["frames"]), s=16, color=col, zorder=3)
        name = f"submap {s['index']}" if s["index"] is not None else "warm-up (replaced)"
        labels.append(f"{name}\n{len(s['frames'])} views, {s['seconds']:.2f} s")
        y += 1
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(-1, run.n); ax.set_xlabel("frame")
    ax.grid(axis="x", alpha=0.4)
    for e in t["align"]:
        ya = next(i for i, s in enumerate(t["submaps"]) if s["pass"] == e["a_pass"])
        yb = next(i for i, s in enumerate(t["submaps"]) if s["pass"] == e["b_pass"])
        sa, sb = t["submaps"][e["a_pass"]], t["submaps"][e["b_pass"]]
        adjacent = e["b"] - e["a"] == 1
        x = max(sa["start"], sb["start"]) + 6 if adjacent else min(sa["end"], sb["end"]) - 10
        res = (e["info"] or {}).get("residual_rel")
        col = TEXT if adjacent else TEAL
        ax.annotate("", xy=(x, yb - 0.3), xytext=(x, ya + 0.3), arrowprops=dict(arrowstyle="<->", color=col, lw=1.4))
        ax.text(x + 1.0, (ya + yb) / 2, f"{e['shared']} shared · residual {res:.1%}", fontsize=9, color=col,
                va="center", bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec="none", alpha=0.9))
    ax.set_title("Each submap is one model pass over about 32 frames; submaps that share frames are aligned on them",
                 loc="left")
    save(fig, out/"stage4_spans.png")


def fig_map_colours(run, out):
    m = run.m
    rng = np.random.default_rng(0)
    sel = rng.choice(len(m["xyz"]), size=min(len(m["xyz"]), 1_200_000), replace=False)
    xyz = m["xyz"][sel].astype(np.float64)
    sub = m["submap"][sel]
    col = np.array([rgb(SUBMAP[k]) for k in range(4)], np.uint8)[sub]
    shade = (0.55 + 0.45 * (m["rgb"][sel].mean(1, keepdims=True) / 255)).clip(0, 1)
    col = (col * shade).astype(np.uint8)
    centre = np.median(xyz, 0)
    mid = run.poses[run.n // 2, :3, 3]
    eye = mid + np.array([-7.0, -8.0, -7.0])
    view = View(*look_at(eye, centre), size=(1600, 900))
    img = Image.fromarray(splat(view, xyz, col))
    d = ImageDraw.Draw(img)
    draw_path(d, view, run.poses[:, :3, 3], (255, 255, 255), 3)
    for k, s in enumerate(run.committed):
        d.rectangle([24, 24 + 34 * k, 48, 48 + 34 * k], fill=rgb(SUBMAP[s["index"]]))
        d.text((60, 24 + 34 * k), f"submap {s['index']}: frames {s['start']}–{s['end'] - 1}", fill=rgb(TEXT), font=font(24))
    img.save(out/"stage4_submaps3d.jpg", quality=90)
    print("  wrote stage4_submaps3d.jpg")


def fig_long_context(run, out):
    t = run.t
    w = t["windows"][0]
    sel = w["frames"]
    cv = t["covis"][0] if t["covis"] else None
    cfg = t["config"]["backend"]["long_context"]
    ka, kb = w["submaps"][0], w["submaps"][-1]
    side = {}
    if cv:
        side.update({f: ka for f in cv["frames_a"]})
        side.update({f: kb for f in cv["frames_b"]})
    fig = plt.figure(figsize=(16, 7.4))
    gs = fig.add_gridspec(3, 12, height_ratios=[1, 1, 2.3], hspace=0.25, wspace=0.05)
    for i, f in enumerate(sel[:24]):
        ax = fig.add_subplot(gs[i // 12, i % 12])
        ax.imshow(run.images[f]); ax.set_xticks([]); ax.set_yticks([])
        c = SUBMAP[side[f]] if f in side else STEEL
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(3)
        ax.set_title(str(f), fontsize=9, color=MUTED, pad=2)
    if cv and cv.get("best"):
        for j, f in enumerate(cv["best"]):
            k = (ka, kb)[j]
            ax = fig.add_subplot(gs[2, 6 * j:6 * j + 6])
            ax.imshow(run.images[f]); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"frame {f}, seen only by submap {k}", color=SUBMAP[k])
        fig.text(0.5, 0.02, f"Submap {ka} against submap {kb}: {cv['n_strong']} of {cv['n_pairs']} frame pairs share enough "
                 f"keypoints ({cv['frac_strong']:.1%}); a link needs {cfg['min_covis']:.0%}. Rejected. Above: the window's "
                 f"24 frames, bordered by the submap each was compared for.", ha="center", color=TEXT, fontsize=12)
    save(fig, out/"stage5_long_context.png")


def fig_graph(run, out):
    t = run.t
    S_after = np.array(t["pgo"][-1]["after"]) if t["pgo"] else run.a["submap_poses"]
    S_before = np.array(t["pgo"][-1]["before"]) if t["pgo"] else S_after
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ks = [s["index"] for s in run.committed]
    X = {k: 3.0 * i for i, k in enumerate(ks)}
    res = {(e["a"], e["b"]): (e["info"] or {}).get("residual_rel") for e in t["align"]}

    def arc(a, b, rad, **kw):
        ax.annotate("", xy=(X[b], 0), xytext=(X[a], 0), zorder=2,
                    arrowprops=dict(arrowstyle="-", connectionstyle=f"arc3,rad={rad}", shrinkA=22, shrinkB=22, **kw))
    for a, b in zip(ks[:-1], ks[1:]):
        arc(a, b, 0.0, color=TEXT, lw=3)
        ax.text((X[a] + X[b]) / 2, 0.14, f"{res.get((a, b), 0):.1%}", ha="center", fontsize=10, color=TEXT)
    for e in t["span"]:
        arc(e[0], e[1], -0.45, color=TEAL, lw=2.2, ls="--")
        ax.text((X[e[0]] + X[e[1]]) / 2, 1.08, f"{res.get((e[0], e[1]), 0):.1%}", ha="center", fontsize=10, color=TEAL)
    for w in t["windows"]:
        if not w["edges"]:
            a, b = w["submaps"][0], w["submaps"][-1]
            arc(a, b, 0.32, color=RED, lw=2.2, ls=":")
            cv = t["covis"][0]["frac_strong"] if t["covis"] else None
            ax.text((X[a] + X[b]) / 2, -2.05, f"long-context link {a}–{b} rejected: covisibility {cv:.1%}",
                    ha="center", fontsize=10.5, color=RED)
    for k in ks:
        s = next(s for s in run.committed if s["index"] == k)
        ax.scatter(X[k], 0, s=1300, color=SUBMAP[k], zorder=4, edgecolor=BG, lw=3)
        ax.text(X[k], 0, str(k), ha="center", va="center", fontsize=16, fontweight="bold", color=BG, zorder=5)
        ax.text(X[k], -0.55, f"frames {s['start']}–{s['end'] - 1}", ha="center", fontsize=10, color=MUTED)
    ax.plot([], [], color=TEXT, lw=3, label="adjacent alignment (residual)")
    ax.plot([], [], color=TEAL, lw=2.2, ls="--", label="span alignment (residual)")
    ax.plot([], [], color=RED, lw=2.2, ls=":", label="long-context candidate")
    ax.set_xlim(-1.6, X[ks[-1]] + 1.6); ax.set_ylim(-2.5, 2.9)
    ax.axis("off"); ax.legend(fontsize=10, loc="upper left", ncol=1, frameon=False)
    ax.set_title(f"The pose graph: {len(ks)} nodes, {len(ks) - 1 + len(t['span'])} edges, "
                 f"{len(t['loops'])} loops", loc="left")
    shift = np.linalg.norm(S_after[:, :3] - S_before[:, :3], axis=1)
    scale = np.abs(np.log(S_after[:, 7] / S_before[:, 7]))
    x = np.arange(len(shift))
    bx.bar(x - 0.2, shift / run.path_len * 100, 0.4, color=RUST, label="position change, % of path length")
    bx.bar(x + 0.2, scale * 100, 0.4, color=TEAL, label="scale change, %")
    bx.set_xticks(x); bx.set_xticklabels([f"submap {k}" for k in x])
    bx.grid(axis="y", alpha=0.4); bx.legend(fontsize=10)
    bx.set_title("What the final optimisation changed", loc="left")
    save(fig, out/"stage7_graph.png")


def depth_edges(d, rtol=0.03):
    pad = np.pad(d, 1, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, (3, 3))
    return (win.max((-1, -2)) - win.min((-1, -2))) / np.maximum(np.abs(d), 1e-6) > rtol


def fig_filter(run, out):
    p = example_pass(run)
    s = run.t["submaps"][p]
    i = len(s["frames"]) // 2
    f = s["frames"][i]
    d = run.a[f"pass{p}_depth"][i].astype(np.float32)
    c = run.a[f"pass{p}_conf"][i].astype(np.float32)
    sky = run.a[f"pass{p}_sky"][i]
    low = c < np.quantile(c, 0.2)
    edge = depth_edges(d) & ~low
    sk = sky & ~low & ~edge
    keep = ~(low | edge | sk) & (d > 0)
    img = run.images[f].astype(np.float32)
    over = img * 0.3
    over[keep] = img[keep]
    for mask, col in ((low, VIOLET), (edge, YELLOW), (sk, TEAL)):
        over[mask] = 0.35 * img[mask] + 0.65 * np.array(rgb(col))
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 4.9))
    a.imshow(run.images[f]); a.set_title(f"frame {f}", loc="left")
    b.imshow(over.astype(np.uint8))
    n = d.size
    b.set_title(f"kept {keep.sum() / n:.0%} of pixels", loc="left")
    for col, lab, m in ((VIOLET, "below the 20th confidence percentile", low), (YELLOW, "depth edge", edge), (TEAL, "sky", sk)):
        b.scatter([], [], s=80, color=col, marker="s", label=f"{lab}: {m.sum() / n:.0%}")
    b.legend(loc="lower right", fontsize=10)
    for ax in (a, b):
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, out/"stage8_filter.png")


def fig_final(run, out):
    m = run.m
    rng = np.random.default_rng(1)
    sel = rng.choice(len(m["xyz"]), size=min(len(m["xyz"]), 2_000_000), replace=False)
    xyz, col = m["xyz"][sel].astype(np.float64), m["rgb"][sel]
    pos, fwd = smooth_poses(run.poses, 12)
    views = {"chase": chase_view(pos[0], fwd[0], (1600, 900), back=3.0, up=1.4, ahead=5.0),
             "high": View(*look_at(pos[run.n // 2] + np.array([-6.0, -8.0, -6.0]), np.median(xyz, 0)), size=(1600, 900))}
    for name, view in views.items():
        img = Image.fromarray(splat(view, xyz, col))
        draw_path(ImageDraw.Draw(img), view, run.poses[:, :3, 3], rgb(RUST), 4)
        for r in run.t["frontend"]:
            if r["promoted"]:
                draw_frustum(img, view, run.poses[r["frame"]], run.K, (run.W, run.H), 0.45, rgb(YELLOW), 2)
        img.save(out/f"map_{name}.jpg", quality=90)
        print(f"  wrote map_{name}.jpg")


# ---- the hero film --------------------------------------------------------------------------
def label(d, xy, text, fill, font):
    """Text on a translucent dark backdrop, readable over bright sky."""
    x0, y0, x1, y1 = d.textbbox(xy, text, font=font)
    d.rounded_rectangle((x0 - 8, y0 - 5, x1 + 8, y1 + 6), radius=6, fill=(10, 12, 14, 170))
    d.text(xy, text, fill=fill, font=font)


def film(run, out, size=(1920, 1080), inset_w=560, orbit_s=5.0):
    """The SLAM camera's chase view, full frame, with the input video picture-in-picture; then an orbit."""
    W, H = size
    iw, ih = inset_w, inset_w * 9 // 16
    frames = run.video_frames(iw, ih)
    m = run.m
    rng = np.random.default_rng(2)
    sel = rng.choice(len(m["xyz"]), size=min(len(m["xyz"]), 1_800_000), replace=False)
    xyz, col, src = m["xyz"][sel].astype(np.float64), m["rgb"][sel], m["frame"][sel]
    pos, fwd = smooth_poses(run.poses, 14)
    big, small = font(34), font(26)
    proc = run.t["timing"]["run_s"]
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(run.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out/"film.mp4")]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def render(view, upto, f, texture):
        img = Image.fromarray(splat(view, xyz[src <= upto], col[src <= upto], size_k=0.0075 * view.f, max_size=5))
        d = ImageDraw.Draw(img)
        draw_path(d, view, run.poses[:f + 1, :3, 3], rgb(RUST), 5)
        draw_frustum(img, view, run.poses[f], run.K, (run.W, run.H), 0.55, rgb(YELLOW), 3, texture)
        return img

    def inset(img, frame, caption):
        x, y = 28, H - ih - 28
        d = ImageDraw.Draw(img, "RGBA")
        d.rounded_rectangle((x - 4, y - 4, x + iw + 4, y + ih + 4), radius=10, fill=(10, 12, 14, 255))
        img.paste(Image.fromarray(frame), (x, y))
        label(d, (x + 12, y + 10), caption, (255, 255, 255), small)
        return d

    for f in range(run.n):
        img = render(chase_view(pos[f], fwd[f], size, back=2.9, up=1.35), f, f, Image.fromarray(frames[f]))
        d = inset(img, frames[f], "Input · generated video")
        label(d, (28, 24), "AMB3R-SLAM · SLAM camera, its path and the map", (255, 255, 255), big)
        label(d, (W - 470, H - 64), f"frame {f:3d} / {run.n - 1} · {f / run.fps:4.2f} s", (235, 235, 235), small)
        enc.stdin.write(img.tobytes())
    centre = np.median(xyz, 0)
    n_orbit = int(orbit_s * run.fps)
    for i in range(n_orbit):
        a = 2 * np.pi * i / n_orbit * 0.75 - 0.35 * np.pi
        eye = centre + np.array([9.0 * np.sin(a), -5.5, -9.0 * np.cos(a)])
        img = render(View(*look_at(eye, centre), size=size), run.n, run.n - 1, None)
        d = inset(img, frames[-1], "Input · last frame")
        label(d, (28, 24), "The whole map", (255, 255, 255), big)
        label(d, (W - 560, H - 64), f"{run.n} frames · {proc:.1f} s on one RTX 4090", (220, 220, 220), small)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    print("  wrote film.mp4")


# ---- the 3D viewer's data ---------------------------------------------------------------------
def viewer_data(run, out, max_points=1_000_000):
    """scene.json + points.bin, in three.js axes (y up): x' = x, y' = -y, z' = -z."""
    vd = out/"viewer"
    vd.mkdir(exist_ok=True)
    m, t = run.m, run.t
    rng = np.random.default_rng(3)
    sel = np.sort(rng.choice(len(m["xyz"]), size=min(len(m["xyz"]), max_points), replace=False))
    F = np.diag([1.0, -1.0, -1.0])
    xyz = (m["xyz"][sel].astype(np.float64) @ F).astype(np.float32)
    conf = np.clip(m["conf_rel"][sel].astype(np.float32) / 3.0, 0, 1)
    blobs = [xyz.tobytes(), m["rgb"][sel].astype(np.uint8).tobytes(), m["frame"][sel].astype(np.uint8).tobytes(),
             m["submap"][sel].astype(np.uint8).tobytes(), (conf * 255).round().astype(np.uint8).tobytes()]
    (vd/"points.bin").write_bytes(b"".join(blobs))
    F4 = np.diag([1.0, -1.0, -1.0, 1.0])
    conv = lambda P: [np.round((F4 @ p @ F4).reshape(-1), 5).tolist() for p in P]  # noqa: E731
    track = run.a["frontend_track"].astype(np.float64)
    track_ok = np.isfinite(track).all((1, 2))
    S = np.array(t["pgo"][-1]["after"]) if t["pgo"] else run.a["submap_poses"]
    scene = {
        "count": int(len(sel)), "layout": ["position:f32x3", "color:u8x3", "frame:u8", "submap:u8", "conf:u8"],
        "fps": run.fps, "frames": run.n, "image": [run.W, run.H],
        "K": [float(run.K[0, 0]), float(run.K[1, 1]), float(run.K[0, 2]), float(run.K[1, 2])],
        "poses": conv(run.poses),
        "frontend": [conv([p])[0] if ok else None for p, ok in zip(track, track_ok)],
        "keyframes": [r["frame"] for r in t["frontend"] if r["promoted"]],
        "frontend_views": [r["views"] or [r["frame"]] for r in t["frontend"]],
        "submaps": [{"index": s["index"], "start": s["start"], "end": s["end"], "frames": s["frames"],
                     "seconds": round(s["seconds"], 3), "replaced": s["replaced"], "pass": s["pass"],
                     "colour": SUBMAP[s["index"]] if s["index"] is not None else STEEL} for s in t["submaps"]],
        "passes": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()} for p in t["passes"]
                   if p["model"] == "backend"],
        "edges": ([{"type": "adjacent", "a": k, "b": k + 1} for k in range(len(run.committed) - 1)]
                  + [{"type": "span", "a": e[0], "b": e[1]} for e in t["span"]]
                  + [{"type": "long_context_rejected", "a": w["submaps"][0], "b": w["submaps"][-1],
                      "covis": t["covis"][0]["frac_strong"] if t["covis"] else None} for w in t["windows"]
                     if not w["edges"]]
                  + [{"type": "long_context", "a": e[0], "b": e[1]} for e in t["long_context"]]
                  + [{"type": "loop", "a": e[0], "b": e[1]} for e in t["loops"]]),
        "probes": [{"frame": p["frame"], "match": p["match"]} for p in t["probes"]],
        "submap_poses": [[round(float(v), 5) for v in s] for s in S],
        "timing": {k: round(v, 3) if isinstance(v, float) else v for k, v in t["timing"].items() if k != "stages_s"},
        "stages_s": {k: round(v, 3) for k, v in t["timing"]["stages_s"].items()},
        "counts": t["counts"],
    }
    (vd/"scene.json").write_text(json.dumps(scene, separators=(",", ":")))
    print(f"  wrote viewer data: {len(sel)} points, {(vd/'points.bin').stat().st_size / 1e6:.1f} MB")


def summary(run, out):
    t = run.t
    s = {"frames": t["frames"], "model": t["model"], "timing": t["timing"], "peak_gpu_gb": t["peak_gpu_gb"],
         "counts": t["counts"], "path_length_model_units": run.path_len,
         "keyframes": [r["frame"] for r in t["frontend"] if r["promoted"]],
         "submaps": [{k: s[k] for k in ("index", "start", "end", "seconds", "replaced")} | {"views": len(s["frames"])}
                     for s in t["submaps"]],
         "align": [{"a": e["a"], "b": e["b"], "shared": e["shared"], "residual_rel": (e["info"] or {}).get("residual_rel")}
                   for e in t["align"]],
         "long_context": {"windows": t["windows"], "covis": t["covis"],
                          "min_covis": t["config"]["backend"]["long_context"]["min_covis"]},
         "loops": {"probes": len(t["probes"]), "matches": sum(p["match"] >= 0 for p in t["probes"]),
                   "candidates": len(t["loop_checks"]), "accepted": len(t["loops"])},
         "intrinsics_median": {"fx": float(run.K[0, 0]), "fy": float(run.K[1, 1]), "cx": float(run.K[0, 2]),
                               "cy": float(run.K[1, 2]), "image": [run.W, run.H]}}
    (out/"summary.json").write_text(json.dumps(s, indent=1))
    print("  wrote summary.json")


STEPS = {"input": fig_input, "frames": fig_frames, "frontend": fig_frontend, "submap": fig_submap,
         "submap3d": fig_submap3d, "spans": fig_spans, "colours": fig_map_colours, "long_context": fig_long_context,
         "graph": fig_graph, "filter": fig_filter, "final": fig_final, "viewer": viewer_data, "summary": summary,
         "film": film}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", choices=sorted(STEPS))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = Run(a.trace, a.video)
    for name in a.only or STEPS:
        STEPS[name](run, out)


if __name__ == "__main__":
    main()
