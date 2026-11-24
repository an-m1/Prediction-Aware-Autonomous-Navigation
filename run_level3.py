import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
import numpy as np

from astar import AStarPlanner
from dwb import DWBLocalPlanner
from rrt import RRTPlanner
from mdwb import MDWBLocalPlanner
from lidar_tracker import LidarPedestrianTracker, noisy_observations_from_gt
from pedestrian_motion import evaluate_pedestrians, move_pedestrians
from run_level2 import (CRUISE, HALF, ORIGIN, PLAN_RADIUS, RADIUS, RES, clearance, generate_level2_map_pgm, load_grid, scenario as sample_level2_scenario,)

# 2x2 global-planner x local-controller benchmark combinations
CONFIGS = [("AStar", "DWB"), ("RRT", "DWB"), ("AStar", "MDWB"), ("RRT", "MDWB"),]


def make_scenario_scripts(mode, start_xy, goal_xy, rng):
    """Compute corridor midpoint, unit direction, and normal vectors"""
    sx, sy = start_xy
    gx, gy = goal_xy
    mx, my = 0.5 * (sx + gx), 0.5 * (sy + gy)
    dx, dy = gx - sx, gy - sy
    length = max(math.hypot(dx, dy), 1e-6)
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux

    spd_jitter = rng.uniform(0.55, 0.80)
    pause_jitter = rng.uniform(2.0, 3.5)

    # Build pedestrian waypoint scripts for scenarios 'a', 'b', or 'c'
    if mode == "a":
        # Single crossing pedestrian across the corridor
        return [{"id": "pedestrian_1", "waypoints": [(mx - 3.2 * nx, my - 3.2 * ny), (mx + 3.2 * nx, my + 3.2 * ny)], "speed": spd_jitter, "stop_at_idx": None, "stop_duration": 0.0, "radius": 0.25,}]

    if mode == "b":
        # Pedestrian crosses and stops on the planned corridor
        return [{"id": "pedestrian_1", "waypoints": [(mx - 3.2 * nx, my - 3.2 * ny), (mx, my), (mx + 3.2 * nx, my + 3.2 * ny)], "speed": spd_jitter, "stop_at_idx": 1, "stop_duration": pause_jitter, "radius": 0.25,}]

    # Scenario c: two pedestrians with different speeds, timing, and a pause
    return [{"id": "pedestrian_1","waypoints": [(mx - 3.2 * nx - 0.8 * ux, my - 3.2 * ny - 0.8 * uy), (mx, my - 0.8 * uy), (mx + 3.2 * nx - 0.8 * ux, my + 3.2 * ny - 0.8 * uy),], "speed": rng.uniform(0.60, 0.85), "stop_at_idx": 1, "stop_duration": rng.uniform(1.0, 2.0), "radius": 0.25,}, {"id": "pedestrian_2", "waypoints": [(mx + 3.2 * nx + 0.9 * ux, my + 3.2 * ny + 0.9 * uy), (mx - 3.2 * nx + 0.9 * ux, my - 3.2 * ny + 0.9 * uy),], "speed": rng.uniform(0.50, 0.75), "stop_at_idx": None, "stop_duration": 0.0, "radius": 0.25,},]


def plan_with_budget(planner_name, planner, start_xy, goal_xy, rrt_seed=None, time_budget_s=2.0):
    """Time global planning and reject paths exceeding the wall-clock budget"""
    t0 = time.perf_counter()
    if planner_name == "RRT":
        path = planner.plan(start_xy, goal_xy, seed=rrt_seed)
    else:
        path = planner.plan(start_xy, goal_xy)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if elapsed_ms > time_budget_s * 1000.0:
        return None, elapsed_ms
    return path, elapsed_ms


def path_is_blocked(path, wp_idx, obstacles, margin=RADIUS + 0.25):
    """Check upcoming waypoints for insufficient obstacle clearance"""
    if not path or wp_idx >= len(path):
        return True
    for px, py in path[wp_idx : min(len(path), wp_idx + 12)]:
        if clearance(px, py, obstacles) <= margin:
            return True
    return False


def compute_brier_score(controller, robot_pose, future_peds, horizon_steps=5, step_dt=0.5):
    predictor = getattr(controller, "predictor", controller)
    heatmaps = getattr(controller, "_latest_heatmaps", None)
    if heatmaps is None:
        heatmaps = getattr(predictor, "_latest_heatmaps", None)
    if not heatmaps or not future_peds:
        return None

    rx, ry = float(robot_pose[0]), float(robot_pose[1])
    half_w = predictor.half_window
    g_res = predictor.grid_res
    n_cells = predictor.n_cells
    pred_dt = max(predictor.dt, 1e-6)

    # Compare predicted occupancy grids against future ground-truth pedestrian mask
    scores = []
    for k_idx in range(min(horizon_steps, len(future_peds))):
        peds_at_k = future_peds[k_idx]
        t_ahead = (k_idx + 1) * step_dt
        l_idx = min(max(0, int(round(t_ahead / pred_dt)) - 1), len(heatmaps) - 1)
        occ_grid = heatmaps[l_idx]

        # Build binary ground-truth occupancy mask on the 8m x 8m window
        yy, xx = np.indices((n_cells, n_cells), dtype=float)
        cx = (rx - half_w) + (xx + 0.5) * g_res
        cy = (ry - half_w) + (yy + 0.5) * g_res
        gt_occ = np.zeros((n_cells, n_cells), dtype=float)
        for ped in peds_at_k:
            r_ped = float(ped.get("radius", 0.25))
            inside = ((cx - ped["x"]) ** 2 + (cy - ped["y"]) ** 2) <= (r_ped + 0.15) ** 2
            gt_occ = np.maximum(gt_occ, inside.astype(float))

        scores.append(float(np.mean((occ_grid - gt_occ) ** 2)))

    return float(np.mean(scores)) if scores else None


def _resample_path(path, step=0.1):
    """Densify a polyline path into roughly evenly spaced (x, y) waypoints."""
    if not path or len(path) < 2:
        return list(path) if path else []

    step = max(float(step), 1e-3)
    resampled = [(float(path[0][0]), float(path[0][1]))]

    for i in range(len(path) - 1):
        x0, y0 = float(path[i][0]), float(path[i][1])
        x1, y1 = float(path[i + 1][0]), float(path[i + 1][1])
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len <= 1e-9:
            continue

        n_segs = max(1, int(math.ceil(seg_len / step)))
        for k in range(1, n_segs + 1):
            t = k / n_segs
            resampled.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))

    return resampled


def _pack_result(success, failure_reason, elapsed, path_arr, path, min_clear, errors, ang_accels, init_lat_ms, replan_latencies_ms, ctrl_times_ms, brier_scores, trace):
    return {"metrics": {"success": success, "failure_reason": failure_reason, "completion_time_s": elapsed, "path_length_m": float(np.linalg.norm(np.diff(path_arr, axis=0), axis=1).sum()), "min_clearance_m": max(0.0, min_clear), "tracking_error_rms_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0, "path_jaggedness_rad": _jaggedness(path), "control_smoothness": float(np.mean(ang_accels)) if ang_accels else 0.0, "initial_plan_latency_ms": init_lat_ms, "replan_count": len(replan_latencies_ms), "mean_replan_latency_ms": float(np.mean(replan_latencies_ms)) if replan_latencies_ms else 0.0, "mean_controller_ms": float(np.mean(ctrl_times_ms)) if ctrl_times_ms else 0.0, "mean_brier_score": float(np.mean(brier_scores)) if brier_scores else 0.0,}, "trace": trace,}


def run_single_trial(planner_name, ctrl_name, grid, sc, scripts, rrt_seed, obs_rng, max_time=120.0, dt=0.1, w_risk=25.0, replan_time_budget_s=2.0, gz_world=None, record_trace=False,):
    """Initialize global planner, local controller, and initial path"""
    astar = AStarPlanner(grid, RES, ORIGIN, ORIGIN, robot_radius=PLAN_RADIUS)
    rrt = RRTPlanner(grid, RES, ORIGIN, ORIGIN, robot_radius=PLAN_RADIUS)
    planner = rrt if planner_name == "RRT" else astar

    if ctrl_name == "MDWB":
        controller = MDWBLocalPlanner(grid, RES, ORIGIN, ORIGIN, robot_radius=0.20, max_vel_x=max(0.5, CRUISE), goal_tolerance=0.25, cruise_speed_mps=CRUISE, w_risk=w_risk,)
    else:
        controller = DWBLocalPlanner(grid, RES, ORIGIN, ORIGIN, robot_radius=0.20, max_vel_x=max(0.5, CRUISE), goal_tolerance=0.25, cruise_speed_mps=CRUISE, w_risk=0.0,)

    tracker = LidarPedestrianTracker(grid, RES, ORIGIN, ORIGIN)

    raw_path, init_lat_ms = plan_with_budget(planner_name, planner, sc["start"], sc["goal"], rrt_seed=rrt_seed, time_budget_s=replan_time_budget_s)
    replan_latencies_ms = []
    empty_trace = {"start": list(sc["start"]), "goal": list(sc["goal"]), "obstacles": sc["obstacles"], "global_path": raw_path or [], "steps": []}
    if not raw_path or len(raw_path) < 2:
        return _pack_result(False, "planning_failed", max_time, np.zeros((0, 2)), [], 0.0, [], [], init_lat_ms, [], [], [], empty_trace)

    # Initialize simulation state and metric trackers
    path = _resample_path(raw_path, 0.1)
    path_arr = np.asarray(path, dtype=float)
    heading = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    pose = (sc["start"][0], sc["start"][1], heading)
    vel = (0.0, 0.0)

    wp_idx = 0
    min_clear = float("inf")
    errors = []
    ang_accels = []
    ctrl_times_ms = []
    brier_scores = []
    progress_history = [(0.0, pose[0], pose[1])]
    last_replan_t = -5.0
    trace_steps = []

    steps = int(round(max_time / dt))
    for step in range(steps):
        elapsed = step * dt
        x, y, th = pose
        curr_v, curr_w = vel

        # Advance pedestrian ground truth and optionally sync poses to Gazebo
        gt_peds = evaluate_pedestrians(elapsed, scripts=scripts)
        if gz_world:
            move_pedestrians([(p["id"], p["x"], p["y"]) for p in gt_peds], world=gz_world)

        # Check collisions and goal arrival
        clear = clearance(x, y, sc["obstacles"])
        for ped in gt_peds:
            d_ped = math.hypot(x - ped["x"], y - ped["y"]) - RADIUS - float(ped.get("radius", 0.25))
            clear = min(clear, d_ped)
        min_clear = min(min_clear, clear)

        if record_trace:
            trace_steps.append({"t": elapsed, "x": x, "y": y, "th": th, "peds": [{"id": p["id"], "x": p["x"], "y": p["y"]} for p in gt_peds]})

        trace_obj = {"start": list(sc["start"]), "goal": list(sc["goal"]), "obstacles": sc["obstacles"], "global_path": path, "steps": trace_steps}

        if clear <= 0.0:
            return _pack_result(False, "collision", elapsed, path_arr, path, min_clear, errors, ang_accels, init_lat_ms, replan_latencies_ms, ctrl_times_ms, brier_scores, trace_obj)

        if math.dist((x, y), sc["goal"]) <= controller.goal_tolerance:
            return _pack_result(True, "", elapsed, path_arr, path, min_clear, errors, ang_accels, init_lat_ms, replan_latencies_ms, ctrl_times_ms, brier_scores, trace_obj)

        # Replan globally if robot stalls or path ahead is blocked
        progress_history.append((elapsed, x, y))
        five_sec_ago = [pt for pt in progress_history if elapsed - pt[0] >= 5.0]
        stalled = False
        if five_sec_ago:
            ref_t, ref_x, ref_y = five_sec_ago[-1]
            if math.hypot(x - ref_x, y - ref_y) < 0.10:
                stalled = True

        blocked_ahead = path_is_blocked(path, wp_idx, sc["obstacles"])
        if (stalled or blocked_ahead) and (elapsed - last_replan_t >= 2.0):
            last_replan_t = elapsed
            seed_step = None if rrt_seed is None else rrt_seed + (len(replan_latencies_ms) + 1) * 101
            new_raw, rep_ms = plan_with_budget(planner_name, planner, (x, y), sc["goal"], rrt_seed=seed_step, time_budget_s=replan_time_budget_s)
            replan_latencies_ms.append(rep_ms)
            if not new_raw or len(new_raw) < 2:
                return _pack_result(False, "replan_failed", elapsed, path_arr, path, min_clear, errors, ang_accels, init_lat_ms, replan_latencies_ms, ctrl_times_ms, brier_scores, trace_obj)
            path = _resample_path(new_raw, 0.1)
            path_arr = np.asarray(path, dtype=float)
            wp_idx = 0

        errors.append(float(np.min(np.hypot(path_arr[:, 0] - x, path_arr[:, 1] - y))))

        # Computes velocity command and evaluate prediction accuracy for MDWB
        t_ctrl0 = time.perf_counter()
        if ctrl_name == "MDWB":
            obs = noisy_observations_from_gt(gt_peds, obs_rng)
            cmd_v, cmd_w, info = controller.compute_velocity_commands(pose, vel, path, dt=dt, last_waypoint_idx=wp_idx, pedestrian_observations=obs)
            future_slice = [evaluate_pedestrians(elapsed + (k + 1) * 0.5, scripts=scripts) for k in range(5)]
            bs = compute_brier_score(controller, pose, future_slice, horizon_steps=5, step_dt=0.5)
            if bs is not None:
                brier_scores.append(bs)
        else:
            cmd_v, cmd_w, info = controller.compute_velocity_commands(pose, vel, path, dt=dt, last_waypoint_idx=wp_idx)
        ctrl_times_ms.append((time.perf_counter() - t_ctrl0) * 1000.0)

        # Integrate unicycle dynamics and record control smoothness
        wp_idx = info.get("waypoint_idx", wp_idx)
        ang_accels.append(abs(cmd_w - curr_w) / dt)
        new_th = th + cmd_w * dt
        new_x = x + cmd_v * math.cos(new_th) * dt
        new_y = y + cmd_v * math.sin(new_th) * dt
        pose = (new_x, new_y, new_th)
        vel = (cmd_v, cmd_w)

    trace_obj = {"start": list(sc["start"]), "goal": list(sc["goal"]), "obstacles": sc["obstacles"], "global_path": path, "steps": trace_steps}
    return _pack_result(False, "timeout", max_time, path_arr, path, min_clear, errors, ang_accels, init_lat_ms, replan_latencies_ms, ctrl_times_ms, brier_scores, trace_obj)


def _jaggedness(path):
    """Sum heading changes across consecutive path segments"""
    if not path or len(path) < 3:
        return 0.0
    total = 0.0
    for i in range(len(path) - 2):
        th1 = math.atan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])
        th2 = math.atan2(path[i + 2][1] - path[i + 1][1], path[i + 2][0] - path[i + 1][0])
        total += abs(math.atan2(math.sin(th2 - th1), math.cos(th2 - th1)))
    return total


def main():
    """Run the 2x2 Level 3 evaluation benchmark and save results."""

    # Parse CLI options and ensure output directory exists
    parser = argparse.ArgumentParser(description="Level 3 2x2 A*/RRT x DWB/M-DWB benchmark runner.")
    parser.add_argument("--scenarios", nargs="+", choices=["a", "b", "c"], default=["a", "b"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-time", type=float, default=120.0)
    parser.add_argument("--w-risk", type=float, default=25.0)
    parser.add_argument("--replan-budget-s", type=float, default=2.0)
    parser.add_argument("--gz-world", default=None, help="Set e.g. level3_world to push poses to Gazebo")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--save-traces", action="store_true", help="Save per-step robot and pedestrian traces to JSON for plotting")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pgm_path = out_dir / "_level3_2x2_map.pgm"

    layout_rng = random.Random(args.seed)
    all_rows = []
    all_traces = []

    # Generate map/pedestrian scenarios and run all planner-conroller pairs per trial
    for mode in args.scenarios:
        for t_idx in range(1, args.trials + 1):
            sc = sample_level2_scenario(layout_rng, "moderate")
            generate_level2_map_pgm(pgm_path, sc["obstacles"], resolution=RES)
            grid = load_grid(pgm_path)
            ped_rng = random.Random(args.seed + t_idx * 97 + ord(mode))
            scripts = make_scenario_scripts(mode, sc["start"], sc["goal"], ped_rng)
            rrt_seed = args.seed + t_idx * 7919

            for p_name, c_name in CONFIGS:
                obs_rng = random.Random(args.seed + t_idx * 31 + hash((mode, p_name, c_name)) % 10000)
                out = run_single_trial(p_name, c_name, grid, sc, scripts, rrt_seed=rrt_seed, obs_rng=obs_rng, max_time=args.max_time, w_risk=args.w_risk, replan_time_budget_s=args.replan_budget_s, gz_world=args.gz_world, record_trace=args.save_traces,)
                res = out["metrics"]
                row = {"scenario": mode, "trial": t_idx, "planner": p_name, "controller": c_name, "config": f"{p_name}+{c_name}", **res,}
                all_rows.append(row)
                if args.save_traces:
                    all_traces.append({"scenario": mode, "trial": t_idx, "planner": p_name, "controller": c_name, "config": f"{p_name}+{c_name}", "success": res["success"], "failure_reason": res["failure_reason"], **out["trace"]})
        # Print per-scenario success rate across all configurations
        for p_name, c_name in CONFIGS:
            sub = [r for r in all_rows if r["scenario"] == mode and r["planner"] == p_name and r["controller"] == c_name]
            rate = 100.0 * sum(r["success"] for r in sub) / max(len(sub), 1)
            print(f"[SCENARIO {mode.upper()}] {p_name}+{c_name}: success={rate:.1f}% ({len(sub)} trials)")

    # Save raw trial rows to CSV and JSON
    csv_path = out_dir / "level3_2x2_results.csv"
    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    json_path = out_dir / "level3_2x2_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": vars(args), "rows": all_rows}, f, indent=2)

    if args.save_traces:
        traces_path = out_dir / "level3_2x2_traces.json"
        with open(traces_path, "w", encoding="utf-8") as f:
            json.dump(all_traces, f, indent=2)

    print(f"[SAVED] {csv_path} and {json_path}")


if __name__ == "__main__":
    main()