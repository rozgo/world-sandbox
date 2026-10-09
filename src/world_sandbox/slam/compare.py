"""The front-end's live estimate against the final path, for two traced runs of the same video:
before and after a change to the pipeline (here, the hand-off fix in amb3r-slam).

    uv run --locked python -m world_sandbox.slam.compare \
        --before outputs/synthetic_slam/junkyard_v1/trace --after outputs/synthetic_slam/junkyard_v1/trace_fix \
        --json docs/synthetic_slam/results/junkyard_v1_reanchor_fix.json --fig previews/synthetic_slam/junkyard_v1_fix

Writes the numbers as JSON and `stage2_handoff_fix.png`: the live scale, the live step against the
final step, and the live and final paths from above.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from world_sandbox.slam.figures import GRID, MUTED, PANEL, RUST, STEEL, TEAL, TEXT, YELLOW, save

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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


class Run:
    def __init__(self, d):
        d = Path(d)
        self.t = json.loads((d/"trace.json").read_text())
        a = self.a = np.load(d/"arrays.npz")
        self.live = a["frontend_track"][:, :3, 3].astype(float)      # as the front-end returned it
        self.final = a["poses"][:, :3, 3].astype(float)
        self.scale = {r["frame"]: r["scale"] for r in self.t["frontend"] if r["scale"]}
        self.handoffs = [r["after_frame"] for r in self.t["reanchors"]]
        self.first = self.handoffs[0] + 1                            # first frame tracked after a hand-off
        self.idx = np.arange(self.first, len(self.live))
        self.path = float(np.linalg.norm(np.diff(self.final, axis=0), axis=1).sum())
        self.step_live = np.linalg.norm(np.diff(self.live, axis=0), axis=1)
        self.step_final = np.linalg.norm(np.diff(self.final, axis=0), axis=1)

    def table_at(self, frame):
        """Replay, for code before the fix, the poses the front-end fitted ``frame`` against:
        each hand-off overwrote its submap's frames with backend poses (the first submap and its
        warm-ups at the identity, later submaps at their chained Sim(3), S_odo), and left the
        rest as the front-end had placed them."""
        def apply(S, p):
            x, y, z, w = S[3:7]
            R = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                          [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                          [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
            return S[:3] + S[7] * R @ p
        table = {}
        for f in range(frame):
            table[f] = ("front-end", self.live[f])
            for k, r in enumerate(self.t["reanchors"]):
                if r["after_frame"] != f:
                    continue
                sm = self.t["submaps"][k]
                P = self.a[f"pass{k}_poses"][:, :3, 3].astype(float)
                S = self.a["S_odo"][sm["index"]] if sm["index"] else None
                for i, fr in enumerate(r["frames"]):
                    table[fr] = ("backend", P[i] if S is None else apply(S, P[i]))
        rec = next(r for r in self.t["frontend"] if r["frame"] == frame)
        refs = [v for v in rec["views"] if v != frame]
        pairs = [{"frames": [u, v], "written_by": [table[u][0], table[v][0]],
                  "apart_in_table": round(float(np.linalg.norm(table[u][1] - table[v][1])), 3),
                  "apart_on_final_path": round(float(np.linalg.norm(self.final[u] - self.final[v])), 3)}
                 for i, u in enumerate(refs) for v in refs[i + 1:]]
        return {"frame": frame, "fitted_against": refs, "scale_before": round(self.scale[frame - 1], 2),
                "scale_after": round(self.scale[frame], 2), "pairs": pairs}

    def metrics(self):
        i = self.idx
        s, R, tr = umeyama(self.live[i], self.final[i])
        fitted = np.linalg.norm((s * (R @ self.live[i].T)).T + tr - self.final[i], axis=1)
        raw = np.linalg.norm(self.live[i] - self.final[i], axis=1)
        j = i[:-1]                                                    # steps between tracked frames
        ratio = self.step_live[j] / np.maximum(self.step_final[j], 1e-9)
        # Between hand-offs: the live step against the final step, each stretch of tracked frames
        # (the first frame after a hand-off, where the live pose snaps back, reported apart).
        bounds = [h + 1 for h in self.handoffs] + [len(self.live)]
        stretches, snaps = [], []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            fr = np.arange(lo + 1, min(hi, len(self.live)))
            if lo < len(self.live) and lo >= 1:
                snaps.append({"frame": int(lo), "live_step_over_final": round(float(self.step_live[lo - 1] / max(self.step_final[lo - 1], 1e-9)), 1)})
            if len(fr) >= 3:
                r = self.step_live[fr - 1] / np.maximum(self.step_final[fr - 1], 1e-9)
                stretches.append({"frames": [int(fr[0]), int(fr[-1])], "median": round(float(np.median(r)), 2),
                                  "max": round(float(r.max()), 2)})
        sc = [self.scale[f] for f in i if f in self.scale]
        jumps = [self.scale[f] / self.scale[f - 1] for f in i if f in self.scale and f - 1 in self.scale]
        return {
            "upstream": self.t.get("upstream", {"commit": "5418465"}),
            "run_s": round(self.t["timing"]["run_s"], 2),
            "frames_compared": [int(i[0]), int(i[-1])],
            "final_path_length_model_units": round(self.path, 3),
            "live_path_length_over_final": round(float(self.step_live[j].sum() / self.step_final[j].sum()), 2),
            "live_step_over_final_step": {"median": round(float(np.median(ratio)), 2), "max": round(float(ratio.max()), 1)},
            "live_step_over_final_step_between_handoffs": stretches,
            "live_step_over_final_step_first_frame_after_handoff": snaps,
            "live_vs_final_as_returned_max_pct_of_path": round(float(raw.max() / self.path * 100), 1),
            "live_vs_final_after_sim3_fit_pct_of_path": {"rms": round(float(np.sqrt((fitted ** 2).mean()) / self.path * 100), 2),
                                                         "max": round(float(fitted.max() / self.path * 100), 2)},
            "frontend_scale": {"range": [round(min(sc), 2), round(max(sc), 2)],
                               "largest_jump_between_frames": round(float(max(max(jumps), 1 / min(jumps))), 2),
                               "at_frames": {str(f): round(self.scale[f], 2) for f in (32, 64, 65, 96, 128, 144)}},
            "handoff_sim3": [{"after_frame": c["after_frame"], "scale": round(c["scale"], 3), "shift": round(c["shift"], 3)}
                             for c in self.t.get("converts", [])],
        }


def figure(before, after, out):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    ax = axes[0]
    for run, col, name in ((before, STEEL, "before the fix"), (after, TEAL, "after the fix")):
        ax.semilogy(*zip(*sorted(run.scale.items())), color=col, lw=2, label=name)
    for f in after.handoffs:
        ax.axvline(f, color=YELLOW, lw=1, alpha=0.6)
    ax.plot([], [], color=YELLOW, lw=1, label="backend hand-off")
    ax.set_title("Front-end scale, live", loc="left")
    ax.set_xlabel("frame"); ax.grid(alpha=0.5, which="both"); ax.legend(fontsize=10, loc="upper left")

    ax = axes[1]
    for run, col, name in ((before, STEEL, "before"), (after, TEAL, "after")):
        j = run.idx[:-1]
        ax.semilogy(j + 1, run.step_live[j] / np.maximum(run.step_final[j], 1e-9), color=col, lw=1.8, label=name)
    ax.axhline(1, color=MUTED, lw=1.2, ls="--")
    ax.set_title("Live step / final step, per frame (1 = agrees)", loc="left")
    ax.set_xlabel("frame"); ax.grid(alpha=0.5, which="both"); ax.legend(fontsize=10, loc="upper left")

    ax = axes[2]
    i = after.idx
    ax.plot(after.final[:, 0], after.final[:, 2], color=RUST, lw=3, label="final path")
    ax.plot(after.live[i, 0], after.live[i, 2], color=YELLOW, lw=1.6, ls="--", label="live, after the fix")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("After: live and final path, from above", loc="left")
    ax.set_xlabel("x"); ax.set_ylabel("z, forward"); ax.grid(alpha=0.5); ax.legend(fontsize=10, loc="lower left")
    for a in axes:
        a.set_facecolor(PANEL)
        for s in a.spines.values():
            s.set_color(GRID)
        a.tick_params(colors=MUTED)
        a.title.set_color(TEXT)
    fig.tight_layout()
    save(fig, Path(out)/"stage2_handoff_fix.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--fig", required=True)
    a = ap.parse_args()
    before, after = Run(a.before), Run(a.after)
    s, R, tr = umeyama(before.final, after.final)
    moved = np.linalg.norm((s * (R @ before.final.T)).T + tr - after.final, axis=1)
    res = {
        "what": "the front-end's live estimate (pose returned when each frame was tracked) against the final "
                "trajectory, for the same video before and after the hand-off fix; from the first frame tracked "
                "after the first backend hand-off",
        "before_trace": a.before, "after_trace": a.after,
        "before": before.metrics(), "after": after.metrics(),
        "before_fits_across_two_worlds": [before.table_at(f) for f in (32, 64, 65, 96, 128)],
        "final_paths_differ_after_sim3_fit_rms_pct_of_path": round(float(np.sqrt((moved ** 2).mean()) / after.path * 100), 3),
    }
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    Path(a.fig).mkdir(parents=True, exist_ok=True)
    figure(before, after, a.fig)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
