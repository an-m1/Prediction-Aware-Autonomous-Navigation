#!/usr/bin/env python3
"""Level 2 randomized multi-obstacle A*/RRT + DWB evaluation.

Runs a reproducible kinematic evaluation using the same planner and controller
implementations as Level 1. With --ros2, publishes scenarios and trajectories
to the Level 2 Python map visualizer. Outputs paired trials, sensitivity sweeps,
and repeated-seed RRT runs as CSV/JSON in the current directory.
"""
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

HALF = 6.0
RES = 0.05
ORIGIN = -6.0
RADIUS = 0.18
PLAN_RADIUS = 0.28
CRUISE = 0.4
CLUTTER = {"moderate": (3, 4), "dense": (5, 6)}
# These box dimensions match obstacle_1..obstacle_10 in level2_world.sdf.
GAZEBO_OBSTACLE_SIZES = ((1.4, 1.0), (0.9, 1.5), (1.6, 0.9),
                         (1.0, 1.8), (1.4, 1.1), (1.0, 1.3),
                         (0.75, 1.0), (1.2, 0.8), (1.8, 1.2), (1.5, 1.6))


def generate_level2_map_pgm(pgm_path, obstacles, resolution=RES):
    """Rasterize axis-aligned obstacle boxes into a ROS-convention PGM."""
    n = round(2 * HALF / resolution)
    grid = np.full((n, n), 255, dtype=np.uint8)
    wall = max(1, round(0.1 / resolution))
    grid[:wall, :] = grid[-wall:, :] = 0
    grid[:, :wall] = grid[:, -wall:] = 0
    yy, xx = np.indices(grid.shape, dtype=float)
    wx = ORIGIN + (xx + 0.5) * resolution
    wy = ORIGIN + (yy + 0.5) * resolution
    for ob in obstacles:
        c, s = math.cos(ob["yaw"]), math.sin(ob["yaw"])
        dx, dy = wx - ob["x"], wy - ob["y"]
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        grid[(np.abs(lx) <= ob["w"] / 2) & (np.abs(ly) <= ob["h"] / 2)] = 0
    grid = np.flipud(grid)
    with open(pgm_path, "wb") as f:
        f.write(f"P5\n{n} {n}\n255\n".encode())
        f.write(grid.tobytes())


def load_grid(path):
    with open(path, "rb") as f:
        f.readline()
        dims = f.readline().split()
        while dims[0].startswith(b"#"):
            dims = f.readline().split()
        w, h = map(int, dims)
        f.readline()
        return np.flipud(np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w))


def point_clear(x, y, obstacles, margin=0.5):
    if max(abs(x), abs(y)) > HALF - 0.8:
        return False
    for ob in obstacles:
        c, s = math.cos(ob["yaw"]), math.sin(ob["yaw"])
        dx, dy = x - ob["x"], y - ob["y"]
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        if abs(lx) < ob["w"] / 2 + margin and abs(ly) < ob["h"] / 2 + margin:
            return False
    return True


def scenario(rng, clutter):
    count = rng.randint(*CLUTTER[clutter])
    selected = rng.sample(list(enumerate(GAZEBO_OBSTACLE_SIZES, start=1)), count)
    obstacles = []
    # Rejection sampling keeps obstacles separated enough for meaningful routes.
    for model_id, (width, height) in selected:
        for _attempt in range(500):
            ob = {"x": rng.uniform(-4.2, 4.2), "y": rng.uniform(-4.2, 4.2),
                  "w": width, "h": height, "yaw": rng.uniform(-math.pi / 2, math.pi / 2),
                  "model_id": model_id}
            if all(math.hypot(ob["x"] - old["x"], ob["y"] - old["y"]) >
                   0.5 * math.hypot(ob["w"] + old["w"], ob["h"] + old["h"]) + 0.4
                   for old in obstacles):
                obstacles.append(ob)
                break
        else:
            # Preserve the randomly selected count on unusually crowded draws;
            # overlapping boxes are still represented separately in the map.
            obstacles.append({"x": rng.uniform(-4.2, 4.2), "y": rng.uniform(-4.2, 4.2),
                              "w": width, "h": height,
                              "yaw": rng.uniform(-math.pi / 2, math.pi / 2), "model_id": model_id})
    for _ in range(5000):
        start = (rng.uniform(-4.8, 4.8), rng.uniform(-4.8, 4.8))
        goal = (rng.uniform(-4.8, 4.8), rng.uniform(-4.8, 4.8))
        if math.dist(start, goal) >= 6 and point_clear(*start, obstacles) and point_clear(*goal, obstacles):
            return {"start": start, "goal": goal, "obstacles": obstacles}
    raise RuntimeError(f"Could not sample a feasible {clutter} layout")


def clearance(x, y, obstacles):
    d = min(HALF - abs(x), HALF - abs(y))
    for ob in obstacles:
        c, s = math.cos(ob["yaw"]), math.sin(ob["yaw"])
        dx, dy = x - ob["x"], y - ob["y"]
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        qx, qy = max(abs(lx) - ob["w"] / 2, 0), max(abs(ly) - ob["h"] / 2, 0)
        dist = math.hypot(qx, qy)
        if qx == 0 and qy == 0:
            dist = -min(ob["w"] / 2 - abs(lx), ob["h"] / 2 - abs(ly))
        d = min(d, dist)
    return d - RADIUS


def execute(name, planner, controller, sc, seed, max_time=120,
            planned_path=None, planning_latency=None, visualizer=None, gazebo_mover=None):
    if planned_path is None:
        t0 = time.perf_counter()
        path = planner.plan(sc["start"], sc["goal"], seed=seed) if name == "RRT" else planner.plan(sc["start"], sc["goal"])
        latency = (time.perf_counter() - t0) * 1000
    else:
        path, latency = planned_path, planning_latency
    base = {"success": False, "failure_reason": "planning_failed", "planning_latency_ms": latency,
            "completion_time_s": max_time, "path_length_m": 0.0, "min_clearance_m": 0.0,
            "tracking_error_rms_m": 0.0}
    if not path or len(path) < 2:
        return base
    path = planner.resample_path(path, 0.1)
    arr = np.asarray(path)
    length = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
    heading = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    pose, vel = (*sc["start"], heading), (0.0, 0.0)
    if gazebo_mover:
        gazebo_mover.set_model_pose("autonomous_vehicle", pose[0], pose[1], 0.0,
                                    pose[2], world_name="level2_world")
    if visualizer:
        visualizer.publish_pose(*pose)
    min_clear, errors, idx = float("inf"), [], 0
    steps = int(max_time / 0.1)
    for step in range(steps):
        x, y, th = pose
        clear = clearance(x, y, sc["obstacles"])
        min_clear = min(min_clear, clear)
        if clear <= 0:
            base.update(failure_reason="collision", min_clearance_m=min_clear)
            return base
        if math.dist((x, y), sc["goal"]) <= controller.goal_tolerance:
            base.update(success=True, failure_reason="", completion_time_s=step * 0.1,
                        min_clearance_m=min_clear,
                        tracking_error_rms_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0)
            return base
        errors.append(float(np.min(np.hypot(arr[:, 0] - x, arr[:, 1] - y))))
        v, w, info = controller.compute_velocity_commands(pose, vel, path, dt=0.1, last_waypoint_idx=idx)
        idx = info.get("waypoint_idx", idx)
        th += w * 0.1
        pose, vel = (x + v * math.cos(th) * 0.1, y + v * math.sin(th) * 0.1, th), (v, w)
        if visualizer:
            visualizer.publish_pose(*pose)
        if gazebo_mover:
            gazebo_mover.set_model_pose("autonomous_vehicle", pose[0], pose[1], 0.0,
                                        pose[2], world_name="level2_world")
            time.sleep(0.1)
    base.update(failure_reason="timeout", min_clearance_m=min_clear)
    return base


def build(grid, resolution=RES, robot_radius=PLAN_RADIUS):
    return (AStarPlanner(grid, resolution, ORIGIN, ORIGIN, robot_radius),
            RRTPlanner(grid, resolution, ORIGIN, ORIGIN, robot_radius),
            DWBLocalPlanner(grid, resolution, ORIGIN, ORIGIN, robot_radius=0.20,
                            goal_tolerance=0.25, cruise_speed_mps=CRUISE))


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(pairs):
    """Paired success comparison and metric distributions, with optional SciPy tests."""
    summary = {}
    for clutter in CLUTTER:
        group = [r for r in pairs if r["clutter"] == clutter]
        aok = [r["astar_success"] for r in group]
        rok = [r["rrt_success"] for r in group]
        discordant_a = sum(a and not b for a, b in zip(aok, rok))
        discordant_r = sum(not a and b for a, b in zip(aok, rok))
        try:
            from scipy.stats import binomtest, wilcoxon
            mcnemar_p = float(binomtest(min(discordant_a, discordant_r),
                                        discordant_a + discordant_r, 0.5).pvalue) if discordant_a + discordant_r else 1.0
        except ImportError:
            mcnemar_p = None
        metrics = {}
        for key in ("path_length_m", "completion_time_s", "planning_latency_ms", "min_clearance_m", "tracking_error_rms_m"):
            av = np.array([r[f"astar_{key}"] for r in group if r["astar_success"]], dtype=float)
            rv = np.array([r[f"rrt_{key}"] for r in group if r["rrt_success"]], dtype=float)
            both = [r for r in group if r["astar_success"] and r["rrt_success"]]
            try:
                from scipy.stats import wilcoxon
                p = float(wilcoxon([r[f"astar_{key}"] for r in both], [r[f"rrt_{key}"] for r in both]).pvalue) if both else None
            except (ImportError, ValueError):
                p = None
            metrics[key] = {"astar_median": float(np.median(av)) if av.size else None,
                            "astar_iqr": float(np.percentile(av,75)-np.percentile(av,25)) if av.size else None,
                            "rrt_median": float(np.median(rv)) if rv.size else None,
                            "rrt_iqr": float(np.percentile(rv,75)-np.percentile(rv,25)) if rv.size else None,
                            "paired_wilcoxon_p": p, "paired_n": len(both)}
        summary[clutter] = {"trials": len(group), "astar_success_rate": float(np.mean(aok)) if group else 0.0,
                            "rrt_success_rate": float(np.mean(rok)) if group else 0.0,
                            "mcnemar_exact_p": mcnemar_p, "metrics": metrics,
                            "failure_counts": {
                                "astar": {reason: sum(r["astar_failure_reason"] == reason for r in group)
                                          for reason in ("planning_failed", "collision", "timeout")},
                                "rrt": {reason: sum(r["rrt_failure_reason"] == reason for r in group)
                                        for reason in ("planning_failed", "collision", "timeout")},
                            }}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=30, help="paired trials per clutter level")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-time", type=float, default=120)
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--ros2", action="store_true",
                    help="Publish live scenario and robot updates to level2_visualizer.launch.py.")
    args = ap.parse_args()
    visualizer = None
    gazebo_mover = None
    if args.ros2:
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from std_msgs.msg import String

            class Level2Publisher:
                def __init__(self):
                    self.node = rclpy.create_node("level2_trial_visualization")
                    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
                    self.scenario_pub = self.node.create_publisher(
                        String, "/level2/visualization/scenario", qos)
                    self.pose_pub = self.node.create_publisher(
                        PoseStamped, "/level2/visualization/robot_pose", 10)

                def publish_scenario(self, trial_index, clutter, sc, astar_path,
                                     rrt_path, astar_edges, rrt_edges):
                    def points(items):
                        return [[round(float(x), 4), round(float(y), 4)] for x, y in (items or [])]
                    def edges(items):
                        return [[points([a])[0], points([b])[0]] for a, b in items[:4000]]
                    message = String()
                    message.data = json.dumps({
                        "trial": trial_index, "clutter": clutter, "arena_half": HALF,
                        "start": points([sc["start"]])[0], "goal": points([sc["goal"]])[0],
                        "obstacles": sc["obstacles"], "astar_path": points(astar_path),
                        "rrt_path": points(rrt_path), "astar_edges": edges(astar_edges),
                        "rrt_edges": edges(rrt_edges),
                    }, separators=(",", ":"))
                    self.scenario_pub.publish(message)
                    rclpy.spin_once(self.node, timeout_sec=0.05)

                def publish_pose(self, x, y, yaw):
                    msg = PoseStamped()
                    msg.header.frame_id = "world"
                    msg.header.stamp = self.node.get_clock().now().to_msg()
                    msg.pose.position.x, msg.pose.position.y = float(x), float(y)
                    msg.pose.orientation.z = math.sin(yaw / 2)
                    msg.pose.orientation.w = math.cos(yaw / 2)
                    self.pose_pub.publish(msg)
                    rclpy.spin_once(self.node, timeout_sec=0.0)

            rclpy.init()
            visualizer = Level2Publisher()
            from run_level1 import Ros2NavigationExecutor
            gazebo_mover = Ros2NavigationExecutor()
            print("[INFO] Level 2 trial rendering is connected to Gazebo. Start the Python map with "
                  "ros2 launch vehicle_description level2_visualizer.launch.py")
        except ImportError as exc:
            raise SystemExit("ROS 2 Python packages are unavailable. Source install/setup.bash first.") from exc
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, sensitivity, repeats = [], [], []
    meta = {"seed": args.seed, "paired_trials_per_clutter": args.trials,
            "obstacle_count_range": [3, 6], "clutter": {}}
    scenario_rng = random.Random(args.seed)
    for clutter in CLUTTER:
        paired = []
        for i in range(args.trials):
            sc = scenario(scenario_rng, clutter)
            pgm = out / "_level2_eval_map.pgm"
            generate_level2_map_pgm(pgm, sc["obstacles"])
            grid = load_grid(pgm)
            a, r, d = build(grid)
            t0 = time.perf_counter()
            path_a = a.plan(sc["start"], sc["goal"])
            latency_a = (time.perf_counter() - t0) * 1000
            edges_a = list(a.last_explored_edges)
            rrt_seed = args.seed + i * 7919
            t0 = time.perf_counter()
            path_r = r.plan(sc["start"], sc["goal"], seed=rrt_seed)
            latency_r = (time.perf_counter() - t0) * 1000
            edges_r = list(r.last_explored_edges)
            if visualizer:
                visualizer.publish_scenario(i + 1, clutter, sc, path_a, path_r, edges_a, edges_r)
            if gazebo_mover:
                gazebo_mover.set_model_pose("start_marker", *sc["start"], 0.005,
                                            world_name="level2_world")
                gazebo_mover.set_model_pose("goal_marker", *sc["goal"], 0.005,
                                            world_name="level2_world")
                active_ids = {ob["model_id"] for ob in sc["obstacles"]}
                for obstacle in sc["obstacles"]:
                    gazebo_mover.set_model_pose(
                        f"obstacle_{obstacle['model_id']}", obstacle["x"], obstacle["y"],
                        0.5, obstacle["yaw"], world_name="level2_world")
                for model_id in range(1, len(GAZEBO_OBSTACLE_SIZES) + 1):
                    if model_id not in active_ids:
                        gazebo_mover.set_model_pose(f"obstacle_{model_id}", 30.0 + model_id, 30.0,
                                                    0.5, world_name="level2_world")
                yaw = math.atan2(sc["goal"][1] - sc["start"][1],
                                 sc["goal"][0] - sc["start"][0])
                gazebo_mover.set_model_pose("autonomous_vehicle", *sc["start"], 0.0, yaw,
                                            world_name="level2_world")
            ra = execute("AStar", a, d, sc, None, args.max_time,
                         path_a, latency_a, visualizer, gazebo_mover)
            if gazebo_mover:
                gazebo_mover.set_model_pose("autonomous_vehicle", *sc["start"], 0.0, yaw,
                                            world_name="level2_world")
            rr = execute("RRT", r, d, sc, rrt_seed, args.max_time,
                         path_r, latency_r, visualizer, gazebo_mover)
            pair = {"clutter": clutter, "trial": i + 1, "seed": args.seed,
                    "obstacle_count": len(sc["obstacles"]),
                    "start": sc["start"], "goal": sc["goal"], "obstacles": sc["obstacles"]}
            for prefix, result in (("astar", ra), ("rrt", rr)):
                pair.update({f"{prefix}_{k}": v for k, v in result.items()})
            paired.append(pair)
            rows.append(pair)
            print(f"{clutter:8s} {i+1:02d}/{args.trials}: A*={'ok' if ra['success'] else ra['failure_reason']} "
                  f"RRT={'ok' if rr['success'] else rr['failure_reason']}")
            print(f"[SENSITIVITY] {clutter} layout {i + 1}: evaluating RRT settings and A* resolutions")
            _, sensitivity_rrt, _ = build(grid)
            for step in (0.25, 0.4, 0.6):
                for bias in (0.05, 0.10, 0.20):
                    for budget in (1500, 3000, 6000):
                        t0 = time.perf_counter()
                        path = sensitivity_rrt.plan(sc["start"], sc["goal"], max_iter=budget,
                                       step_size=step, goal_bias=bias, seed=args.seed + i)
                        sensitivity.append({"clutter": clutter, "trial": i + 1, "planner": "RRT",
                                            "step_size": step, "goal_bias": bias, "iterations": budget,
                                            "planning_latency_ms": (time.perf_counter()-t0)*1000,
                                            "success": bool(path)})
            for resolution in (0.05, 0.10, 0.20):
                if resolution == RES:
                    g = grid
                else:
                    temp = out / "_level2_sensitivity.pgm"
                    generate_level2_map_pgm(temp, sc["obstacles"], resolution)
                    g = load_grid(temp)
                ap2, _, _ = build(g, resolution, PLAN_RADIUS)
                p = ap2.plan(sc["start"], sc["goal"])
                sensitivity.append({"clutter": clutter, "trial": i + 1, "planner": "AStar",
                                    "grid_resolution_m": resolution, "success": bool(p),
                                    "path_length_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if p else None})
        meta["clutter"][clutter] = {"trials": len(paired),
            "astar_success_rate": float(np.mean([x["astar_success"] for x in paired])) if paired else 0,
            "rrt_success_rate": float(np.mean([x["rrt_success"] for x in paired])) if paired else 0,
            "paired_trials": paired}
        # Ten seeds on up to ten fixed layouts per clutter level.
        for layout in paired[:10]:
            sc = {"start": layout["start"], "goal": layout["goal"], "obstacles": layout["obstacles"]}
            eval_map = out / "_level2_eval_map.pgm"
            generate_level2_map_pgm(eval_map, sc["obstacles"])
            _, rp, dc = build(load_grid(eval_map))
            for repeat_seed in range(10):
                if repeat_seed == 0:
                    print(f"[REPEATED SEEDS] {clutter} layout {layout['trial']}: 10 RRT seeds")
                result = execute("RRT", rp, dc, sc, args.seed + repeat_seed, args.max_time)
                repeats.append({"clutter": clutter, "layout_trial": layout["trial"],
                                "repeat_seed": args.seed + repeat_seed, **result})
    write_rows(out / "level2_results.csv", rows)
    write_rows(out / "level2_sensitivity.csv", sensitivity)
    write_rows(out / "level2_rrt_repeated_seeds.csv", repeats)
    meta["statistical_summary"] = summarize(rows)
    variability = {}
    for clutter in CLUTTER:
        groups = {}
        for record in repeats:
            if record["clutter"] == clutter:
                groups.setdefault(record["layout_trial"], []).append(record)
        variability[clutter] = {
            "layouts": len(groups),
            "per_layout": {
                str(layout): {
                    "success_rate": float(np.mean([r["success"] for r in records])),
                    "completion_time_sd_s": float(np.std([r["completion_time_s"] for r in records])),
                    "path_length_sd_m": float(np.std([r["path_length_m"] for r in records])),
                    "planning_latency_sd_ms": float(np.std([r["planning_latency_ms"] for r in records])),
                }
                for layout, records in groups.items()
            },
        }
    meta["rrt_repeated_seed_variability"] = variability
    with open(out / "level2_results.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote results to {out.resolve()}")
    if visualizer:
        visualizer.node.destroy_node()
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
