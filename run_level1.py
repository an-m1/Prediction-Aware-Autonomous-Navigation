#!/usr/bin/env python3
"""
Level 1: Single Static Obstacle Navigation Evaluation
Prediction-Aware Autonomous Navigation (EECE 5550)

Pipeline:
- Global Planners: A* (grid-based deterministic) vs RRT (sampling-based stochastic with logged seed)
- Local Controller: Nav2 DWB (Dynamic Window Approach)
- Success Definition: Robot reaches within 0.25 m of goal within 120 s without colliding.
- 30 Paired Trials: Start/goal pairs crossing the central obstacle in a 12m x 12m arena.
  * Fixed 2m x 2m obstacle with per-trial center jitter and rotation.

  * Start/goal positions varied so the straight line between them always crosses the obstacle.
- Markers: flat green cylinder = start pad, flat red cylinder = goal pad.
  Both are thin (h=0.01m) so the robot drives over them without getting stuck.
- Metrics Logged: Success rate, path length, completion time, path jaggedness,
  control smoothness, planning latency, minimum clearance, tracking error.
- Statistics: Wilcoxon signed-rank test, McNemar test, Medians/IQR, 95% Bootstrap CI.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import numpy as np

from astar import AStarPlanner
from dwb import DWBLocalPlanner
from rrt import RRTPlanner


def wait_for_next_trial(trial_index: int, total_trials: int):
    """Keep the completed scene visible until the operator presses x."""
    if trial_index >= total_trials:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive GUI evaluation requires a terminal for the x key prompt.")

    print(f"\n[PAUSED] Trial {trial_index} is complete. Press x to set up trial {trial_index + 1}.")
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        previous = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while sys.stdin.read(1).lower() != "x":
                pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)
    except ImportError:
        while input("Type x then Enter to continue: ").strip().lower() != "x":
            pass


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def generate_level1_map_pgm(
    pgm_path: str,
    obs_cx: float, obs_cy: float,
    obs_w: float, obs_h: float, obs_yaw: float = 0.0,
    resolution: float = 0.05,
    arena_half: float = 6.0,
    origin_x: float = -6.0,
    origin_y: float = -6.0,
):
    """Generate a trial occupancy map matching the obstacle's rotated box pose."""
    n_cells = int(round(2.0 * arena_half / resolution))
    grid = np.full((n_cells, n_cells), 255, dtype=np.uint8)

    # Arena perimeter walls (0.1 m thick)
    wall_cells = max(1, int(round(0.1 / resolution)))
    grid[:wall_cells, :] = 0
    grid[-wall_cells:, :] = 0
    grid[:, :wall_cells] = 0
    grid[:, -wall_cells:] = 0

    # Rasterize the rotated box in the world frame.
    yy, xx = np.indices((n_cells, n_cells), dtype=float)
    wx = origin_x + (xx + 0.5) * resolution - obs_cx
    wy = origin_y + (yy + 0.5) * resolution - obs_cy
    c, sn = math.cos(obs_yaw), math.sin(obs_yaw)
    local_x = c * wx + sn * wy
    local_y = -sn * wx + c * wy
    grid[(np.abs(local_x) <= obs_w / 2.0) & (np.abs(local_y) <= obs_h / 2.0)] = 0

    # Flip vertically for standard ROS 2 PGM convention (row 0 = y_min)
    grid = np.flipud(grid)

    # Write PGM (P5 binary)
    with open(pgm_path, "wb") as f:
        f.write(f"P5\n{n_cells} {n_cells}\n255\n".encode())
        f.write(grid.tobytes())

    return grid, n_cells, n_cells


def write_level1_map_yaml(
    yaml_path: str,
    pgm_path: str,
    resolution: float = 0.05,
    origin_x: float = -6.0,
    origin_y: float = -6.0,
):
    with open(yaml_path, "w") as f:
        f.write(f"image: {os.path.basename(pgm_path)}\n")
        f.write(f"resolution: {resolution}\n")
        f.write(f"origin: [{origin_x}, {origin_y}, 0.0]\n")
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")


def load_map_yaml(yaml_path="level1_map.yaml"):
    """Parses YAML metadata for level1_map."""
    info = {"image": "level1_map.pgm", "resolution": 0.05, "origin": [-6.0, -6.0, 0.0]}
    if not os.path.exists(yaml_path):
        return info
    with open(yaml_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key, val = key.strip(), val.strip()
                if key == "image":
                    info["image"] = val
                elif key == "resolution":
                    info["resolution"] = float(val)
                elif key == "origin":
                    val = val.strip("[]")
                    info["origin"] = [float(x.strip()) for x in val.split(",")]
    return info


def load_pgm(pgm_path="level1_map.pgm"):
    """Loads PGM file and flips vertically to match standard ROS 2 OccupancyGrid frame."""
    with open(pgm_path, "rb") as f:
        header = f.readline().decode().strip()
        line = f.readline().decode()
        while line.startswith("#"):
            line = f.readline().decode()
        dims = line.strip().split()
        width, height = int(dims[0]), int(dims[1])
        line = f.readline().decode()
        while line.startswith("#"):
            line = f.readline().decode()
        maxval = int(line.strip())
        raw_data = f.read()

    arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width))
    # Flip vertically so row 0 corresponds to y_min (origin_y)
    return np.flipud(arr), width, height


# ---------------------------------------------------------------------------
# Level 1 scenario generation (with per-trial obstacle jitter)
# ---------------------------------------------------------------------------

ARENA_HALF = 6.0  # metres; arena is 12m x 12m
ROBOT_CRUISE_SPEED_MPS = 0.4  # Shared moving speed for both A* and RRT runs.


def sample_obstacle(rng: random.Random):
    """
    Returns (cx, cy, width, height) for the trial obstacle.
    - Centre jittered ±0.5m from (0, 0).
    - Width and height sampled in [1.5, 2.5] m independently.
    """
    cx = rng.uniform(-0.5, 0.5)
    cy = rng.uniform(-0.5, 0.5)
    yaw = rng.uniform(-math.pi / 4.0, math.pi / 4.0)
    return cx, cy, 2.0, 2.0, yaw


def line_intersects_box(x1, y1, x2, y2, bx_min, bx_max, by_min, by_max):
    """Checks whether line segment (x1,y1)→(x2,y2) crosses the axis-aligned box."""
    steps = 120
    for i in range(steps + 1):
        t = i / float(steps)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        if bx_min <= x <= bx_max and by_min <= y <= by_max:
            return True
    return False


def generate_level1_scenarios(num_trials=30, seed=12345):
    """
    Generates num_trials paired (start, goal, obstacle_params) configurations.

    For each trial:
      1. Sample obstacle pose jitter and rotation.
      2. Sample start/goal so the straight line crosses the obstacle.
         - Alternates between South→North and West→East approach directions.
         - Start/goal stay >= 1.5m away from obstacle bounding box edges and >= 1.0m from walls.
      3. Rejects layouts where the straight-line Euclidean distance is < 5.0m.

    Returns list of dicts:
      { 'start': (sx, sy), 'goal': (gx, gy),
        'obs_cx': float, 'obs_cy': float,
        'obs_w': float, 'obs_h': float }
    """
    rng = random.Random(seed)
    trials = []
    attempts = 0
    margin_wall = 1.0  # keep start/goal ≥ 1m from arena walls

    while len(trials) < num_trials and attempts < 20000:
        attempts += 1
        obs_cx, obs_cy, obs_w, obs_h, obs_yaw = sample_obstacle(rng)
        extent = math.hypot(obs_w / 2.0, obs_h / 2.0)
        bx_min, bx_max = obs_cx - extent, obs_cx + extent
        by_min, by_max = obs_cy - extent, obs_cy + extent

        # Keep start/goal outside obstacle + 1.5m clearance and inside arena
        safe_x_lo = -ARENA_HALF + margin_wall
        safe_x_hi =  ARENA_HALF - margin_wall
        safe_y_lo = -ARENA_HALF + margin_wall
        safe_y_hi =  ARENA_HALF - margin_wall

        if rng.random() < 0.5:
            # South → North: start in southern half below obstacle, goal in northern half above
            sy_hi = min(by_min - 1.5, -1.0)
            gy_lo = max(by_max + 1.5,  1.0)
            if sy_hi < safe_y_lo + 0.5 or gy_lo > safe_y_hi - 0.5:
                continue
            sx = round(rng.uniform(safe_x_lo, safe_x_hi), 2)
            sy = round(rng.uniform(safe_y_lo, sy_hi), 2)
            gx = round(rng.uniform(safe_x_lo, safe_x_hi), 2)
            gy = round(rng.uniform(gy_lo, safe_y_hi), 2)
        else:
            # West → East
            sx_hi = min(bx_min - 1.5, -1.0)
            gx_lo = max(bx_max + 1.5,  1.0)
            if sx_hi < safe_x_lo + 0.5 or gx_lo > safe_x_hi - 0.5:
                continue
            sx = round(rng.uniform(safe_x_lo, sx_hi), 2)
            sy = round(rng.uniform(safe_y_lo, safe_y_hi), 2)
            gx = round(rng.uniform(gx_lo, safe_x_hi), 2)
            gy = round(rng.uniform(safe_y_lo, safe_y_hi), 2)

        # Must cross central obstacle bounding box
        if not line_intersects_box(sx, sy, gx, gy, bx_min, bx_max, by_min, by_max):
            continue

        # Euclidean distance filter
        if math.hypot(gx - sx, gy - sy) < 5.0:
            continue

        trials.append({
            "start":  (sx, sy),
            "goal":   (gx, gy),
            "obs_cx": round(obs_cx, 3),
            "obs_cy": round(obs_cy, 3),
            "obs_w":  round(obs_w, 3),
            "obs_h":  round(obs_h, 3),
            "obs_yaw": round(obs_yaw, 4),
        })

    if len(trials) < num_trials:
        print(f"[WARN] Only generated {len(trials)}/{num_trials} valid scenarios after {attempts} attempts.")

    return trials


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_path_jaggedness(path):
    """Total absolute heading change along planned path (radians)."""
    if not path or len(path) < 3:
        return 0.0
    jaggedness = 0.0
    for i in range(len(path) - 2):
        p0, p1, p2 = path[i], path[i + 1], path[i + 2]
        th1 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        th2 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        dth = abs(math.atan2(math.sin(th2 - th1), math.cos(th2 - th1)))
        jaggedness += dth
    return jaggedness


def compute_distance_to_obstacle(x, y, bx_min, bx_max, by_min, by_max, arena_half=ARENA_HALF):
    """Distance from (x,y) to closest obstacle boundary (central box or arena walls)."""
    dx = max(bx_min - x, 0.0, x - bx_max)
    dy = max(by_min - y, 0.0, y - by_max)
    d_box = math.hypot(dx, dy)
    if bx_min <= x <= bx_max and by_min <= y <= by_max:
        d_box = 0.0
    d_wall = min(arena_half - abs(x), arena_half - abs(y))
    return min(d_box, d_wall)


# ---------------------------------------------------------------------------
# Single trial execution
# ---------------------------------------------------------------------------

def run_trial(
    planner_name, planner, controller,
    start, goal,
    obs_cx, obs_cy, obs_w, obs_h, obs_yaw=0.0,
    rrt_seed=None, max_time=120.0, dt=0.1,
    physical_radius=0.18, ros_executor=None,
    trial_index=None, status_interval_s=5.0,
):
    """
    Executes a single trial:
    1. Global planning (A* or RRT) → logs planning latency.
    2. DWB local controller tracking → logs execution trajectory and metrics.
    Supports live ROS 2 / Gazebo execution if ros_executor is provided.
    """
    bx_min = obs_cx - obs_w / 2.0
    bx_max = obs_cx + obs_w / 2.0
    by_min = obs_cy - obs_h / 2.0
    by_max = obs_cy + obs_h / 2.0
    planner_label = "A*" if planner_name == "AStar" else planner_name
    status_prefix = (
        f"Level 1: trial {trial_index}" if trial_index is not None else "Level 1"
    )

    # 1. Global Planning
    if ros_executor is not None:
        print(f"[STATUS] Planning with {planner_label} ({status_prefix}).")
    t0 = time.time()
    if planner_name == "RRT":
        raw_path = planner.plan(start, goal, seed=rrt_seed)
    else:
        raw_path = planner.plan(start, goal)
    planning_latency_ms = (time.time() - t0) * 1000.0

    if raw_path is None or len(raw_path) < 2:
        if ros_executor is not None:
            print(f"[STATUS] {planner_label}, {status_prefix}: no collision-free global path found.")
        return {
            "success": False,
            "failure_reason": "planning_failed",
            "planning_latency_ms": planning_latency_ms,
            "completion_time_s": max_time,
            "path_length_m": 0.0,
            "path_jaggedness_rad": 0.0,
            "control_smoothness": 0.0,
            "min_clearance_m": 0.0,
            "tracking_error_rms_m": 0.0,
            "trajectory": [],
            "planned_path": [],
        }

    # Resample path into 0.1m-spaced waypoints
    planned_path = planner.resample_path(raw_path, spacing=0.1)
    jaggedness = compute_path_jaggedness(planned_path)

    if ros_executor is not None:
        print(
            f"[STATUS] Currently moving towards goal using {planner_label}+DWB "
            f"at {controller.cruise_speed_mps:.2f} m/s ({status_prefix})."
        )

    # 2. Local Controller Execution (DWB)
    init_dx = planned_path[1][0] - start[0]
    init_dy = planned_path[1][1] - start[1]
    init_th = math.atan2(init_dy, init_dx)

    pose = (start[0], start[1], init_th)
    vel  = (0.0, 0.0)

    trajectory       = [(pose[0], pose[1])]
    velocities       = [vel]
    angular_accels   = []
    cross_track_errors = []

    elapsed_time  = 0.0
    success       = False
    failure_reason = ""
    min_clearance = float("inf")
    path_arr      = np.array(planned_path)
    last_wp_idx   = 0
    next_status_s  = status_interval_s

    steps = int(max_time / dt)

    for step in range(steps):
        if ros_executor is not None:
            ros_executor.spin_once()
            world_pose = ros_executor.get_world_pose()
            if world_pose is not None:
                pose = world_pose
                vel = ros_executor.current_vel

        curr_x, curr_y, curr_th = pose
        curr_v, curr_w = vel
        goal_distance = math.hypot(goal[0] - curr_x, goal[1] - curr_y)

        if ros_executor is not None and elapsed_time >= next_status_s:
            print(
                f"[STATUS] {planner_label}+DWB, {status_prefix}: "
                f"{goal_distance:.2f} m remaining after {elapsed_time:.1f} s."
            )
            next_status_s += status_interval_s

        # Obstacle clearance (physical footprint)
        # Measure clearance in the obstacle's local frame, including its yaw.
        dx, dy = curr_x - obs_cx, curr_y - obs_cy
        c, sn = math.cos(obs_yaw), math.sin(obs_yaw)
        local_x, local_y = c * dx + sn * dy, -sn * dx + c * dy
        qx = max(abs(local_x) - obs_w / 2.0, 0.0)
        qy = max(abs(local_y) - obs_h / 2.0, 0.0)
        dist_to_obs = math.hypot(qx, qy)
        if qx == 0.0 and qy == 0.0:
            dist_to_obs = -min(obs_w / 2.0 - abs(local_x), obs_h / 2.0 - abs(local_y))
        dist_to_wall = min(ARENA_HALF - abs(curr_x), ARENA_HALF - abs(curr_y))
        clearance = min(dist_to_obs, dist_to_wall) - physical_radius
        if clearance < min_clearance:
            min_clearance = clearance

        if clearance <= 0.0:
            success = False
            failure_reason = "collision"
            if ros_executor is not None:
                ros_executor.stop_robot()
                print(f"[STATUS] {planner_label}+DWB, {status_prefix}: stopped for collision.")
            break

        # Goal reach (within 0.25 m)
        if goal_distance <= controller.goal_tolerance:
            success = True
            if ros_executor is not None:
                ros_executor.stop_robot()
                print(
                    f"[STATUS] {planner_label}+DWB, {status_prefix}: goal reached "
                    f"in {elapsed_time:.1f} s."
                )
            break

        # Cross-track error
        dists_to_path = np.hypot(path_arr[:, 0] - curr_x, path_arr[:, 1] - curr_y)
        cross_track_errors.append(np.min(dists_to_path))

        # DWB velocity command
        cmd_v, cmd_w, info = controller.compute_velocity_commands(
            pose, vel, planned_path, dt=dt, last_waypoint_idx=last_wp_idx
        )
        last_wp_idx = info.get("waypoint_idx", last_wp_idx)

        ang_acc = abs(cmd_w - curr_w) / dt
        angular_accels.append(ang_acc)

        # Advance the same differential-drive model that DWB evaluated, then
        # render that resulting robot pose in Gazebo. This keeps the visible
        # robot synchronized with the controller even when wheel odometry is
        # reset by Gazebo after a teleport between trials.
        new_th = curr_th + cmd_w * dt
        new_x  = curr_x + cmd_v * math.cos(new_th) * dt
        new_y  = curr_y + cmd_v * math.sin(new_th) * dt
        pose = (new_x, new_y, new_th)
        vel  = (cmd_v, cmd_w)
        if ros_executor is not None:
            ros_executor.render_robot_pose(new_x, new_y, new_th, cmd_v, cmd_w)
            time.sleep(dt)

        trajectory.append((pose[0], pose[1]))
        velocities.append(vel)
        elapsed_time += dt

    if not success and failure_reason == "":
        failure_reason = "timeout"
        if ros_executor is not None:
            ros_executor.stop_robot()
            print(
                f"[STATUS] {planner_label}+DWB, {status_prefix}: "
                f"timed out after {max_time:.1f} s."
            )

    path_length = sum(
        math.hypot(trajectory[k + 1][0] - trajectory[k][0],
                   trajectory[k + 1][1] - trajectory[k][1])
        for k in range(len(trajectory) - 1)
    )
    smoothness  = float(np.mean(angular_accels)) if angular_accels else 0.0
    rms_error   = float(np.sqrt(np.mean(np.square(cross_track_errors)))) if cross_track_errors else 0.0

    return {
        "success":             success,
        "failure_reason":      failure_reason,
        "planning_latency_ms": planning_latency_ms,
        "completion_time_s":   elapsed_time,
        "path_length_m":       path_length,
        "path_jaggedness_rad": jaggedness,
        "control_smoothness":  smoothness,
        "min_clearance_m":     max(0.0, min_clearance),
        "tracking_error_rms_m": rms_error,
        "trajectory":          trajectory,
        "planned_path":        planned_path,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(x, y):
    """Two-sided Wilcoxon signed-rank test (pure Python / NumPy)."""
    diffs = np.array(x) - np.array(y)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n < 5:
        return float("nan"), float("nan")

    abs_diffs = np.abs(diffs)
    ranks = np.empty(n, dtype=float)
    sorted_indices = np.argsort(abs_diffs)

    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs_diffs[sorted_indices[j]] == abs_diffs[sorted_indices[j + 1]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = rank
        i = j + 1

    w_plus  = np.sum(ranks[diffs > 0])
    w_minus = np.sum(ranks[diffs < 0])
    w_stat  = min(w_plus, w_minus)

    mu    = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma < 1e-6:
        return w_stat, 1.0

    z     = (w_stat - mu) / sigma
    p_val = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
    return float(w_stat), float(min(1.0, max(0.0, p_val)))


def mcnemar_test(successes_a, successes_b):
    """McNemar's test for paired binary outcomes."""
    b = sum(1 for sa, sb in zip(successes_a, successes_b) if sa and not sb)
    c = sum(1 for sa, sb in zip(successes_a, successes_b) if not sa and sb)
    if b + c == 0:
        return 0.0, 1.0
    chi2  = ((abs(b - c) - 1.0) ** 2) / float(b + c)
    p_val = math.erfc(math.sqrt(chi2) / math.sqrt(2.0))
    return float(chi2), float(min(1.0, max(0.0, p_val)))


def bootstrap_diff_ci(diffs, n_boot=2000, alpha=0.05):
    """95% bootstrap confidence interval of mean difference."""
    diffs = np.array(diffs)
    if len(diffs) < 2:
        return 0.0, 0.0
    boot_means = [
        np.mean(np.random.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_boot)
    ]
    low  = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return low, high


# ---------------------------------------------------------------------------
# ROS 2 / Gazebo executor
# ---------------------------------------------------------------------------

class Ros2NavigationExecutor:
    """
    Manages ROS 2 communications with Gazebo Sim.
    Topics: /odom (in), /cmd_vel (out); Gazebo set_pose service for trial resets.
    Uses gz service calls to reposition the start_marker, goal_marker,
    and central_obstacle models per trial.
    """

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry

        self.node    = Node("level1_navigation_node")
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_sub = self.node.create_subscription(
            Odometry, "/odom", self._odom_cb, 10
        )
        # Diff-drive odometry is local to the robot's spawn/reset frame.  Keep
        # that local pose separate from the world pose used by the planners.
        self.odom_pose = None
        self.current_vel  = (0.0, 0.0)
        self.pose_service_available = None
        self.world_start = None
        self.world_start_yaw = 0.0
        self.odom_origin = None
        self.last_world_pose = None
        self.controlled_world_pose = None

    def _odom_cb(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q  = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.odom_pose = (px, py, yaw)
        self.current_vel  = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def spin_once(self):
        import rclpy
        rclpy.spin_once(self.node, timeout_sec=0.01)

    def get_world_pose(self):
        """Convert local wheel odometry into the current trial's world frame."""
        if self.controlled_world_pose is not None:
            return self.controlled_world_pose
        if self.world_start is None:
            return None
        if self.odom_pose is None:
            return self.last_world_pose or (
                self.world_start[0], self.world_start[1], self.world_start_yaw)
        if self.odom_origin is None:
            self.odom_origin = self.odom_pose
        ox, oy, oyaw = self.odom_pose
        origin_x, origin_y, origin_yaw = self.odom_origin
        ox -= origin_x
        oy -= origin_y
        oyaw -= origin_yaw
        c, s = math.cos(self.world_start_yaw), math.sin(self.world_start_yaw)
        candidate = (
            self.world_start[0] + c * ox - s * oy,
            self.world_start[1] + s * ox + c * oy,
            self.world_start_yaw + oyaw,
        )
        # A teleport may cause DiffDrive odometry to jump back to zero on its
        # next update. Rebase rather than treating that jump as robot motion.
        if (self.last_world_pose is not None and
                math.hypot(candidate[0] - self.last_world_pose[0],
                           candidate[1] - self.last_world_pose[1]) > 0.30):
            self.odom_origin = self.odom_pose
            candidate = self.last_world_pose
        self.last_world_pose = candidate
        return candidate

    def stop_robot(self):
        from geometry_msgs.msg import Twist
        # Send several zero commands so no queued velocity carries into a reset.
        for _ in range(3):
            self.cmd_pub.publish(Twist())

    def send_velocity(self, v, w):
        from geometry_msgs.msg import Twist
        t = Twist()
        t.linear.x  = float(v)
        t.angular.z = float(w)
        self.cmd_pub.publish(t)

    def render_robot_pose(self, x: float, y: float, yaw: float,
                          linear_velocity: float, angular_velocity: float,
                          world_name: str = "level1_world"):
        """Render DWB's next differential-drive pose in the Gazebo GUI."""
        self.set_model_pose("autonomous_vehicle", x, y, 0.0, yaw, world_name)
        self.controlled_world_pose = (x, y, yaw)
        self.last_world_pose = self.controlled_world_pose
        self.current_vel = (linear_velocity, angular_velocity)

    def set_model_pose(self, model_name: str, x: float, y: float, z: float,
                       yaw: float = 0.0, world_name: str = "level1_world"):
        """Move a Gazebo model, using ROS service or Gazebo Transport CLI."""
        import subprocess

        service = f"/world/{world_name}/set_pose"
        if self.pose_service_available is not False:
            try:
                from ros_gz_interfaces.srv import SetEntityPose
                from geometry_msgs.msg import Point, Quaternion
                import rclpy

                client = self.node.create_client(SetEntityPose, service)
                if not client.wait_for_service(timeout_sec=1.5):
                    self.pose_service_available = False
                    raise RuntimeError(f"ROS bridge service unavailable: {service}")
                req = SetEntityPose.Request()
                req.entity.name = model_name
                req.entity.type = 2
                req.pose.position = Point(x=float(x), y=float(y), z=float(z))
                req.pose.orientation = Quaternion(
                    x=0.0, y=0.0,
                    z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))
                future = client.call_async(req)
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
                if future.done() and future.result() is not None and future.result().success:
                    self.pose_service_available = True
                    return
                self.pose_service_available = False
                raise RuntimeError(f"ROS bridge rejected pose update for {model_name}")
            except Exception as ros_error:
                self.pose_service_available = False
                print(f"[WARN] {ros_error}; trying Gazebo Transport set_pose.")

        quat_z = math.sin(yaw / 2.0)
        quat_w = math.cos(yaw / 2.0)
        request = (
            f'name: "{model_name}" '
            f'position: {{x: {x:.6f}, y: {y:.6f}, z: {z:.6f}}} '
            f'orientation: {{x: 0, y: 0, z: {quat_z:.9f}, w: {quat_w:.9f}}}'
        )
        try:
            result = subprocess.run(
                [
                    "gz", "service",
                    "-s", f"{service}/blocking",
                    "--reqtype", "gz.msgs.Pose",
                    "--reptype", "gz.msgs.Boolean",
                    "--timeout", "5000",
                    "--req", request,
                ],
                capture_output=True, text=True, timeout=15.0,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to set pose for {model_name}: {exc}") from exc
        response = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0 or "data: true" not in response.lower():
            details = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Gazebo rejected pose update for {model_name}: {details}")

    def teleport_robot(self, sx: float, sy: float, yaw: float = 0.0,
                       world_name: str = "level1_world"):
        """Teleport the robot and establish the local-odom to world transform."""
        self.stop_robot()
        time.sleep(0.2)
        self.set_model_pose("autonomous_vehicle", sx, sy, 0.0, yaw, world_name)
        # Gazebo's DiffDrive odometry does not publish the teleported world pose.
        # Its next message is the displacement from this reset point.
        self.world_start = (sx, sy)
        self.world_start_yaw = yaw
        self.odom_pose = None
        self.odom_origin = None
        self.last_world_pose = (sx, sy, yaw)
        self.controlled_world_pose = self.last_world_pose
        self.current_vel = (0.0, 0.0)
        self.stop_robot()
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Level 1: Single Static Obstacle Navigation Evaluation"
    )
    parser.add_argument("--trials",   type=int,   default=30,    help="Paired evaluation trials (default: 30)")
    parser.add_argument("--max_time", type=float, default=120.0, help="Time budget per trial in seconds (default: 120.0)")
    parser.add_argument("--seed",     type=int,   default=12345, help="Scenario generation random seed (default: 12345)")
    parser.add_argument("--ros2", action="store_true", help="Run the Gazebo GUI evaluation")
    args = parser.parse_args()
    if not args.ros2:
        parser.error("This project runs Level 1 in the live Gazebo GUI; pass --ros2.")

    try:
        import rclpy
        print("[INFO] ROS 2 (rclpy) detected. Initializing ROS 2 environment...")
        rclpy.init()
        ros_exec = Ros2NavigationExecutor()
        print("[INFO] ROS 2 node initialized. Listening to /odom and publishing /cmd_vel.")
    except ImportError as exc:
        raise SystemExit("ROS 2 Python packages are required. Source install/setup.bash and rerun.") from exc

    print("=" * 80)
    print(" LEVEL 1 GUI EVALUATION: A* + DWB vs. RRT + DWB")
    print("=" * 80)
    print(f"[CONFIG] Trials: {args.trials}, Max time: {args.max_time}s, Seed: {args.seed}")
    print("[CONFIG] Center obstacle: fixed 2.0x2.0m, position jitter +/-0.5m, varying yaw")
    print(f"[CONFIG] Start/Goal markers: flat cylinders (r=0.4m, h=0.01m) in Gazebo")
    print(f"[CONFIG] Robot cruise speed: {ROBOT_CRUISE_SPEED_MPS:.2f} m/s (A* and RRT)")

    # Physical / planner radii
    physical_robot_radius   = 0.18
    global_planning_radius  = 0.28
    local_controller_radius = 0.20

    # 1. Generate 30 Paired Scenarios (each with its own obstacle config)
    trials_meta = generate_level1_scenarios(num_trials=args.trials, seed=args.seed)
    print(f"[SCENARIOS] Generated {len(trials_meta)} unique trials.")

    results_a   = []
    results_rrt = []

    print("-" * 80)
    print(f"{'#':<4} {'Start':<16} {'Goal':<16} {'Obstacle pose':<20} | "
          f"{'A* Time':<9} {'A* Len':<8} {'A* Lat':<8} | "
          f"{'RRT Time':<9} {'RRT Len':<8} {'RRT Lat':<8}")
    print("-" * 80)

    for i, trial in enumerate(trials_meta, start=1):
        start  = trial["start"]
        goal   = trial["goal"]
        obs_cx = trial["obs_cx"]
        obs_cy = trial["obs_cy"]
        obs_w  = trial["obs_w"]
        obs_h  = trial["obs_h"]
        rrt_seed = 1000 + i * 37

        # 2. Regenerate occupancy map for this trial's obstacle geometry
        generate_level1_map_pgm(
            "level1_map.pgm",
            obs_cx, obs_cy, obs_w, obs_h, trial["obs_yaw"],
        )
        write_level1_map_yaml("level1_map.yaml", "level1_map.pgm")

        # Reload map and rebuild planners / controller for this trial
        grid, width, height = load_pgm("level1_map.pgm")
        res = 0.05
        origin_x, origin_y = -6.0, -6.0

        a_planner  = AStarPlanner(grid, res, origin_x, origin_y, robot_radius=global_planning_radius)
        r_planner  = RRTPlanner( grid, res, origin_x, origin_y, robot_radius=global_planning_radius)
        controller = DWBLocalPlanner(
            grid, res, origin_x, origin_y,
            robot_radius=local_controller_radius,
            goal_tolerance=0.25,
            w_path=40.0,
            w_goal=4.0,
            w_align=14.0,
            w_speed=4.0,
            cruise_speed_mps=ROBOT_CRUISE_SPEED_MPS,
        )

        # 3. In ROS 2 mode: reposition Gazebo markers and robot
        if ros_exec is not None:
            # Move start and goal cylinder markers
            ros_exec.set_model_pose("start_marker", start[0], start[1], 0.005)
            ros_exec.set_model_pose("goal_marker",  goal[0],  goal[1],  0.005)
            # Move and rotate the fixed-size obstacle to this trial's pose.
            ros_exec.set_model_pose("central_obstacle", obs_cx, obs_cy, 0.5, trial["obs_yaw"])

        # 4. Run both planners from the exact same physical start pose.
        init_yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
        if ros_exec is not None:
            ros_exec.teleport_robot(start[0], start[1], init_yaw)
        res_a = run_trial(
            "AStar", a_planner, controller, start, goal,
            obs_cx, obs_cy, obs_w, obs_h, trial["obs_yaw"],
            max_time=args.max_time,
            physical_radius=physical_robot_radius,
            ros_executor=ros_exec,
            trial_index=i,
        )
        if ros_exec is not None:
            ros_exec.teleport_robot(start[0], start[1], init_yaw)
        res_r = run_trial(
            "RRT", r_planner, controller, start, goal,
            obs_cx, obs_cy, obs_w, obs_h, trial["obs_yaw"],
            rrt_seed=rrt_seed,
            max_time=args.max_time,
            physical_radius=physical_robot_radius,
            ros_executor=ros_exec,
            trial_index=i,
        )

        results_a.append(res_a)
        results_rrt.append(res_r)

        str_s   = f"({start[0]:.1f},{start[1]:.1f})"
        str_g   = f"({goal[0]:.1f},{goal[1]:.1f})"
        obs_str = f"{obs_w:.1f}x{obs_h:.1f}@{trial['obs_yaw']:.2f}rad"
        status_a = f"{res_a['completion_time_s']:.1f}s" if res_a["success"] else "FAIL"
        status_r = f"{res_r['completion_time_s']:.1f}s" if res_r["success"] else "FAIL"

        print(
            f"{i:<4} {str_s:<16} {str_g:<16} {obs_str:<18} | "
            f"{status_a:<9} {res_a['path_length_m']:<8.2f} {res_a['planning_latency_ms']:<8.1f} | "
            f"{status_r:<9} {res_r['path_length_m']:<8.2f} {res_r['planning_latency_ms']:<8.1f}"
        )
        wait_for_next_trial(i, len(trials_meta))

    # 5. Aggregate Statistics
    metrics_keys = [
        ("path_length_m",        "Path Length (m)"),
        ("completion_time_s",    "Completion Time (s)"),
        ("path_jaggedness_rad",  "Path Jaggedness (rad)"),
        ("control_smoothness",   "Control Smoothness (rad/s²)"),
        ("planning_latency_ms",  "Planning Latency (ms)"),
        ("min_clearance_m",      "Min Clearance (m)"),
        ("tracking_error_rms_m", "Tracking Error RMS (m)"),
    ]

    success_a = [r["success"] for r in results_a]
    success_r = [r["success"] for r in results_rrt]
    succ_rate_a = 100.0 * sum(success_a) / len(success_a)
    succ_rate_r = 100.0 * sum(success_r) / len(success_r)

    print("=" * 80)
    print(" LEVEL 1 SUMMARY & STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"Goal Reach Success Rate: A*+DWB = {succ_rate_a:.1f}%, RRT+DWB = {succ_rate_r:.1f}%")
    chi2, p_mcnemar = mcnemar_test(success_a, success_r)
    print(f"McNemar Test: chi2 = {chi2:.2f}, p-value = {p_mcnemar:.4f}")
    print("-" * 80)
    print(f"{'Metric':<28} | {'A* (Median ± IQR)':<22} | {'RRT (Median ± IQR)':<22} | {'Wilcoxon p':<10} | {'95% CI'}")
    print("-" * 80)

    stats_summary = {}
    for key, label in metrics_keys:
        vals_a = [r[key] for r in results_a   if r["success"]]
        vals_r = [r[key] for r in results_rrt if r["success"]]
        all_a = [r[key] for r in results_a]
        all_r = [r[key] for r in results_rrt]
        if not vals_a:
            vals_a = all_a
        if not vals_r:
            vals_r = all_r

        med_a = float(np.median(vals_a))
        iqr_a = float(np.percentile(vals_a, 75) - np.percentile(vals_a, 25))
        med_r = float(np.median(vals_r))
        iqr_r = float(np.percentile(vals_r, 75) - np.percentile(vals_r, 25))

        paired_diffs = [
            ra[key] - rb[key]
            for ra, rb in zip(results_a, results_rrt)
            if ra["success"] and rb["success"]
        ]
        w_stat, p_val = wilcoxon_signed_rank(
            [ra[key] for ra, rb in zip(results_a, results_rrt) if ra["success"] and rb["success"]],
            [rb[key] for ra, rb in zip(results_a, results_rrt) if ra["success"] and rb["success"]],
        )
        ci_low, ci_high = bootstrap_diff_ci(paired_diffs)

        stats_summary[key] = {
            "label":          label,
            "astar_median":   med_a,
            "astar_iqr":      iqr_a,
            "rrt_median":     med_r,
            "rrt_iqr":        iqr_r,
            "wilcoxon_stat":  w_stat,
            "wilcoxon_p":     p_val,
            "ci_95":          [ci_low, ci_high],
        }

        p_str = f"{p_val:.4f}" if not math.isnan(p_val) else "N/A"
        print(
            f"{label:<28} | {med_a:8.2f} ± {iqr_a:<8.2f}   | "
            f"{med_r:8.2f} ± {iqr_r:<8.2f}   | {p_str:<10} | [{ci_low:.2f}, {ci_high:.2f}]"
        )

    print("=" * 80)

    # Success Criteria
    print("\nSUCCESS CRITERIA VERIFICATION:")
    crit1 = (succ_rate_a >= 90.0) and (succ_rate_r >= 90.0)
    print(f" 1. Goal-reach rate >= 90%:  {'PASSED' if crit1 else 'FAILED'} "
          f"(A*={succ_rate_a:.1f}%, RRT={succ_rate_r:.1f}%)")
    successful_runs = [r for r in results_a + results_rrt if r["success"]]
    crit2 = bool(successful_runs) and all(r["min_clearance_m"] > 0 for r in successful_runs)
    print(f" 2. Collision-free paths:    {'PASSED' if crit2 else 'FAILED'}")
    crit3 = len(stats_summary) == len(metrics_keys)
    print(f" 3. Baseline metrics logged: {'PASSED' if crit3 else 'FAILED'}")

    # 6. Export Results
    csv_file = "level1_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial",
            "start_x", "start_y", "goal_x", "goal_y",
            "obs_cx", "obs_cy", "obs_w", "obs_h", "obs_yaw",
            "astar_success", "astar_time_s", "astar_length_m", "astar_jaggedness_rad",
            "astar_smoothness", "astar_latency_ms", "astar_clearance_m", "astar_rmse_m",
            "rrt_success", "rrt_time_s", "rrt_length_m", "rrt_jaggedness_rad",
            "rrt_smoothness", "rrt_latency_ms", "rrt_clearance_m", "rrt_rmse_m",
        ])
        for idx, trial in enumerate(trials_meta):
            s, g = trial["start"], trial["goal"]
            ra, rb = results_a[idx], results_rrt[idx]
            writer.writerow([
                idx + 1,
                s[0], s[1], g[0], g[1],
                trial["obs_cx"], trial["obs_cy"], trial["obs_w"], trial["obs_h"], trial["obs_yaw"],
                ra["success"], ra["completion_time_s"], ra["path_length_m"],
                ra["path_jaggedness_rad"], ra["control_smoothness"],
                ra["planning_latency_ms"], ra["min_clearance_m"], ra["tracking_error_rms_m"],
                rb["success"], rb["completion_time_s"], rb["path_length_m"],
                rb["path_jaggedness_rad"], rb["control_smoothness"],
                rb["planning_latency_ms"], rb["min_clearance_m"], rb["tracking_error_rms_m"],
            ])

    json_file = "level1_results.json"
    with open(json_file, "w") as f:
        json.dump({
            "metadata": {
                "level":           1,
                "trials":          len(trials_meta),
                "arena_size_m":    "12 x 12",
                "obstacle":        "fixed 2m x 2m box, center jitter ?0.5m, varying yaw",
                "robot_radius_m":  physical_robot_radius,
                "goal_tolerance_m": 0.25,
                "max_time_s":      args.max_time,
                "seed":            args.seed,
            },
            "success_rates": {
                "astar":      succ_rate_a,
                "rrt":        succ_rate_r,
                "mcnemar_p":  p_mcnemar,
            },
            "statistics":  stats_summary,
            "trials_meta": trials_meta,
        }, f, indent=2)

    print(f"[OUTPUT] Trial metrics: {csv_file}, {json_file}")


if __name__ == "__main__":
    main()
