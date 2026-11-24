import argparse
import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

CONFIGS = [("AStar", "DWB"), ("RRT", "DWB"), ("AStar", "MDWB"), ("RRT", "MDWB"),]


def _wilcoxon_two_sided(x, y):
    """Two-sided Wilcoxon signed-rank test using SciPy if available, else NumPy."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    diffs = x_arr - y_arr
    nonzero = diffs != 0.0
    if np.count_nonzero(nonzero) < 5:
        return None
    try:
        from scipy.stats import wilcoxon
        _, p_val = wilcoxon(x_arr[nonzero], y_arr[nonzero], alternative="two-sided")
        return float(p_val)
    except ImportError:
        d = diffs[nonzero]
        n = len(d)
        abs_d = np.abs(d)
        order = np.argsort(abs_d)
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        w_stat = float(min(np.sum(ranks[d > 0]), np.sum(ranks[d < 0])))
        mu = n * (n + 1) / 4.0
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        if sigma < 1e-12:
            return 1.0
        z = (w_stat - mu) / sigma
        return float(min(1.0, max(0.0, math.erfc(abs(z) / math.sqrt(2.0)))))


def _mcnemar_p(succ_a, succ_b):
    """Exact binomial McNemar p-value on discordant paired outcomes."""
    b = sum(1 for a, b_val in zip(succ_a, succ_b) if bool(a) and not bool(b_val))
    c = sum(1 for a, b_val in zip(succ_a, succ_b) if not bool(a) and bool(b_val))
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n=n_disc, p=0.5).pvalue)
    except ImportError:
        chi2 = ((abs(b - c) - 1.0) ** 2) / float(n_disc)
        return float(min(1.0, max(0.0, math.erfc(math.sqrt(chi2) / math.sqrt(2.0)))))


def _bootstrap_ci_95(diffs, n_boot=2000, seed=12345):
    """95% bootstrap confidence interval for the mean paired difference."""
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = np.mean(arr[idx], axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _holm_adjust(p_values):
    """Holm step-down adjustment for multiple comparisons; returns list aligned with input."""
    m = len(p_values)
    adjusted = [None] * m
    indexed = [(idx, p) for idx, p in enumerate(p_values) if p is not None and math.isfinite(p)]
    indexed.sort(key=lambda item: item[1])
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        factor = m - rank
        running_max = max(running_max, min(1.0, p_val * factor))
        adjusted[orig_idx] = float(running_max)
    return adjusted


def summarize_2x2_stats(rows, alpha=0.05):
    """Compute medians/IQRs, paired tests, 95% bootstrap CIs, and Holm-corrected p-values."""
    metric_keys = ["completion_time_s", "path_length_m", "min_clearance_m", "tracking_error_rms_m", "path_jaggedness_rad", "control_smoothness", "initial_plan_latency_ms", "mean_controller_ms", "mean_brier_score",]
    pair_defs = [("AStar+DWB", "RRT+DWB"), ("AStar+MDWB", "RRT+MDWB"), ("AStar+DWB", "AStar+MDWB"), ("RRT+DWB", "RRT+MDWB"),]

    scenarios = sorted({r["scenario"] for r in rows})
    summary = {"alpha": alpha, "by_scenario": {}}

    for sc_mode in scenarios:
        sc_rows = [r for r in rows if r["scenario"] == sc_mode]
        by_cfg = {f"{p}+{c}": {r["trial"]: r for r in sc_rows if r["config"] == f"{p}+{c}"} for p, c in CONFIGS}

        cfg_stats = {}
        for cfg_name, trial_map in by_cfg.items():
            t_list = list(trial_map.values())
            succ_list = [r for r in t_list if r.get("success")]
            failures = {}
            for r in t_list:
                if not r.get("success"):
                    reason = r.get("failure_reason") or "failed"
                    failures[reason] = failures.get(reason, 0) + 1

            m_dist = {}
            for mk in metric_keys:
                vals = np.asarray([r.get(mk, 0.0) for r in (succ_list or t_list)], dtype=float)
                m_dist[mk] = {"median": float(np.median(vals)) if vals.size else 0.0, "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)) if vals.size else 0.0,}

            cfg_stats[cfg_name] = {"trials": len(t_list), "success_rate": float(np.mean([1.0 if r.get("success") else 0.0 for r in t_list])) if t_list else 0.0, "failure_counts": failures, "metrics": m_dist,}

        # Pairwise McNemar + Wilcoxon tests across key comparisons
        mcnemar_raw = []
        mcnemar_entries = []
        for cfg_1, cfg_2 in pair_defs:
            map_1, map_2 = by_cfg[cfg_1], by_cfg[cfg_2]
            common_trials = sorted(set(map_1.keys()) & set(map_2.keys()))
            s1 = [map_1[t].get("success", False) for t in common_trials]
            s2 = [map_2[t].get("success", False) for t in common_trials]
            p_mc = _mcnemar_p(s1, s2)
            mcnemar_raw.append(p_mc)
            mcnemar_entries.append((f"{cfg_1}_vs_{cfg_2}", p_mc))

        mcnemar_holm = _holm_adjust(mcnemar_raw)
        mcnemar_out = {}
        for (label, p_raw), p_adj in zip(mcnemar_entries, mcnemar_holm):
            mcnemar_out[label] = {"p_raw": p_raw, "p_holm": p_adj, "significant": bool(p_adj is not None and p_adj < alpha),}

        wilcoxon_out = {}
        for mk in metric_keys:
            raw_p_list = []
            pair_records = []
            for idx_p, (cfg_1, cfg_2) in enumerate(pair_defs):
                map_1, map_2 = by_cfg[cfg_1], by_cfg[cfg_2]
                paired_trials = [t for t in sorted(set(map_1.keys()) & set(map_2.keys())) if map_1[t].get("success") and map_2[t].get("success")]
                v1 = [map_1[t].get(mk, 0.0) for t in paired_trials]
                v2 = [map_2[t].get(mk, 0.0) for t in paired_trials]
                diffs = [a - b for a, b in zip(v1, v2)]
                p_w = _wilcoxon_two_sided(v1, v2)
                ci = _bootstrap_ci_95(diffs, seed=12345 + idx_p * 17)
                raw_p_list.append(p_w)
                pair_records.append((f"{cfg_1}_vs_{cfg_2}", {"paired_n": len(paired_trials), "median_diff": float(np.median(diffs)) if diffs else 0.0, "ci_95": ci, "p_raw": p_w,},))
            
            
            adj_p_list = _holm_adjust(raw_p_list)
            mk_dict = {}
            for (comp_key, rec), p_adj in zip(pair_records, adj_p_list):
                rec["p_holm"] = p_adj
                rec["significant"] = bool(p_adj is not None and p_adj < alpha)
                mk_dict[comp_key] = rec
            wilcoxon_out[mk] = mk_dict

        summary["by_scenario"][sc_mode] = {"configs": cfg_stats, "mcnemar": mcnemar_out, "wilcoxon": wilcoxon_out,}

    return summary


def plot_one(trace, out_file):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw static obstacles as solid circles
    for obs in trace.get("obstacles", []):
        if isinstance(obs, dict):
            ox, oy, rad = float(obs["x"]), float(obs["y"]), float(obs.get("radius", 0.25))
        else:
            ox, oy, rad = float(obs[0]), float(obs[1]), float(obs[2])
        ax.add_patch(Circle((ox, oy), rad, color="dimgray", alpha=0.8))

    # Plot start and goal markers
    sx, sy = trace["start"]
    gx, gy = trace["goal"]
    ax.scatter([sx], [sy], c="green", s=80, zorder=5, label="Start")
    ax.scatter([gx], [gy], c="gold", edgecolors="black", s=100, zorder=5, label="Goal")

    # Plot executed robot trajectory
    steps = trace.get("steps", [])
    if steps:
        rx = [s["x"] for s in steps]
        ry = [s["y"] for s in steps]
        ax.plot(rx, ry, c="tab:blue", lw=2.0, label="Robot path")
        if trace.get("failure_reason") == "collision":
            ax.scatter([rx[-1]], [ry[-1]], c="red", marker="x", s=110, zorder=6, label="Collision")

    # Plot pedestrian trajectories by ID
    ped_tracks = {}
    for s in steps:
        for p in s.get("peds", []):
            ped_tracks.setdefault(p["id"], ([], []))
            ped_tracks[p["id"]][0].append(p["x"])
            ped_tracks[p["id"]][1].append(p["y"])

    for pid, (px, py) in ped_tracks.items():
        ax.plot(px, py, "--", lw=1.6, label=f"{pid}")
        ax.scatter([px[0]], [py[0]], s=35)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Scenario {trace['scenario'].upper()} | Trial {trace['trial']} | {trace['config']} (ok={trace['success']})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Level 3 results JSON and/or plot saved trial traces.")
    parser.add_argument("--results", default="level3_2x2_results.json")
    parser.add_argument("--output", default="level3_2x2_stats.json")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--traces", default="level3_2x2_traces.json")
    parser.add_argument("--scenario", choices=["a", "b", "c"], default=None)
    parser.add_argument("--trial", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default="trial_plots")
    args = parser.parse_args()

    if not args.skip_stats and Path(args.results).exists():
        payload = json.load(open(args.results, encoding="utf-8"))
        rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
        summary = summarize_2x2_stats(rows, alpha=args.alpha)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[SAVED] {out_path}")

    if args.scenario is not None and args.trial is not None and Path(args.traces).exists():
        traces = json.load(open(args.traces, encoding="utf-8"))
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        matches = [t for t in traces if t["scenario"] == args.scenario and t["trial"] == args.trial and (args.config is None or t["config"] == args.config)]
        for t in matches:
            fname = out_dir / f"sc_{t['scenario']}_t{t['trial']}_{t['config'].replace('+', '_')}.png"
            plot_one(t, fname)
            print(f"[PLOTTED] {fname}")


if __name__ == "__main__":
    main()