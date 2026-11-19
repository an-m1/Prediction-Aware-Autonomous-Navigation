#!/usr/bin/env python3
"""Level 3 dynamic-obstacle A*/RRT + DWB paired trial evaluation."""

import argparse
import json
import math
import random
import time
from pathlib import Path

from run_level2 import (
    CRUISE, HALF, build, execute, generate_level2_map_pgm, load_grid,
    point_clear, wait_for_next_trial, write_rows,
)


STATIC_OBSTACLES = [
    {"x": -1.7, "y": 3.0, "w": 2.0, "h": 2.0, "yaw": 0.0, "model_id": 1},
    {"x": 0.0, "y": 0.0, "w": 1.6, "h": 1.6, "yaw": 0.0, "model_id": 2},
    {"x": 3.0, "y": -3.0, "w": 2.0, "h": 2.0, "yaw": 0.0, "model_id": 3},
]


def pedestrian_positions(elapsed):
    """Same scripted pedestrian motion used by pedestrian_motion.py."""
    return [
        {"name": "pedestrian_1", "x": -3.0, "y": 2.0 * math.sin(0.3 * elapsed), "radius": 0.25},
        {"name": "pedestrian_2", "x": 3.0 + 1.8 * math.sin(0.35 * elapsed), "y": 3.0, "radius": 0.25},
        {"name": "pedestrian_3", "x": 1.5, "y": -1.5 + 2.5 * math.sin(0.25 * elapsed), "radius": 0.25},
    ]


def sample_scenario(rng):
    for _ in range(5000):
        start = (rng.uniform(-4.8, 4.8), rng.uniform(-4.8, 4.8))
        goal = (rng.uniform(-4.8, 4.8), rng.uniform(-4.8, 4.8))
        if (math.dist(start, goal) >= 6.0
                and point_clear(*start, STATIC_OBSTACLES, margin=0.6)
                and point_clear(*goal, STATIC_OBSTACLES, margin=0.6)):
            return {"start": start, "goal": goal,
                    "obstacles": [dict(obstacle) for obstacle in STATIC_OBSTACLES]}
    raise RuntimeError("Could not generate a valid Level 3 start/goal pair")


class Level3Publisher:
    def __init__(self):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self.rclpy = rclpy
        self.PoseStamped = PoseStamped
        self.String = String
        self.node = rclpy.create_node("level3_trial_visualization")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.scenario_pub = self.node.create_publisher(
            String, "/level3/visualization/scenario", qos)
        self.pose_pub = self.node.create_publisher(
            PoseStamped, "/level3/visualization/robot_pose", 10)
        self.pedestrian_pub = self.node.create_publisher(
            String, "/level3/visualization/pedestrians", 10)

    @staticmethod
    def _points(items):
        return [[round(float(x), 4), round(float(y), 4)] for x, y in (items or [])]

    def publish_scenario(self, trial, scenario, astar_path, rrt_path, astar_edges, rrt_edges):
        def edges(items):
            return [[self._points([a])[0], self._points([b])[0]] for a, b in items[:4000]]
        message = self.String()
        message.data = json.dumps({
            "trial": trial, "arena_half": HALF,
            "start": self._points([scenario["start"]])[0],
            "goal": self._points([scenario["goal"]])[0],
            "obstacles": scenario["obstacles"],
            "astar_path": self._points(astar_path), "rrt_path": self._points(rrt_path),
            "astar_edges": edges(astar_edges), "rrt_edges": edges(rrt_edges),
        }, separators=(",", ":"))
        self.scenario_pub.publish(message)
        self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def publish_pose(self, x, y, yaw):
        message = self.PoseStamped()
        message.header.frame_id = "world"
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.pose.position.x, message.pose.position.y = float(x), float(y)
        message.pose.orientation.z = math.sin(yaw / 2.0)
        message.pose.orientation.w = math.cos(yaw / 2.0)
        self.pose_pub.publish(message)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_pedestrians(self, pedestrians):
        message = self.String()
        message.data = json.dumps(pedestrians, separators=(",", ":"))
        self.pedestrian_pub.publish(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30, help="paired Level 3 trials")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-time", type=float, default=120.0)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--ros2", action="store_true",
                        help="render the robot in Gazebo and publish the Python map visualization")
    args = parser.parse_args()

    visualizer = gazebo_mover = None
    if args.ros2:
        try:
            import rclpy
            from run_level1 import Ros2NavigationExecutor
            rclpy.init()
            visualizer = Level3Publisher()
            gazebo_mover = Ros2NavigationExecutor()
            print("[INFO] Level 3 rendering connected. Start the map with "
                  "ros2 launch vehicle_description level3_visualizer.launch.py")
        except ImportError as exc:
            raise SystemExit("ROS 2 Python packages are unavailable. Source install/setup.bash first.") from exc

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    print("=" * 112)
    print(" LEVEL 3 GUI EVALUATION: A* + DWB vs. RRT + DWB (MOVING PEDESTRIANS)")
    print("=" * 112)
    print(f"[CONFIG] Paired trials: {args.trials}, Max time: {args.max_time}s, Seed: {args.seed}")
    print("[CONFIG] Dynamic obstacles: 3 scripted pedestrians")
    print(f"[CONFIG] Robot cruise speed: {CRUISE:.2f} m/s (A* and RRT)")
    print("-" * 112)
    print(f"{'#':<4} {'Start':<16} {'Goal':<16} {'Ped':<5} | "
          f"{'A* Time':<9} {'A* Len':<8} {'A* Lat':<9} | "
          f"{'RRT Time':<9} {'RRT Len':<8} {'RRT Lat':<9}")
    print("-" * 112)

    for trial in range(1, args.trials + 1):
        scenario = sample_scenario(rng)
        map_path = out / "_level3_eval_map.pgm"
        generate_level2_map_pgm(map_path, scenario["obstacles"])
        grid = load_grid(map_path)
        astar, rrt, controller = build(grid)

        started = time.perf_counter()
        astar_path = astar.plan(scenario["start"], scenario["goal"])
        astar_latency = (time.perf_counter() - started) * 1000.0
        astar_edges = list(astar.last_explored_edges)
        rrt_seed = args.seed + trial * 7919
        started = time.perf_counter()
        rrt_path = rrt.plan(scenario["start"], scenario["goal"], seed=rrt_seed)
        rrt_latency = (time.perf_counter() - started) * 1000.0
        rrt_edges = list(rrt.last_explored_edges)

        if visualizer:
            visualizer.publish_scenario(trial, scenario, astar_path, rrt_path,
                                        astar_edges, rrt_edges)
        initial_yaw = math.atan2(scenario["goal"][1] - scenario["start"][1],
                                 scenario["goal"][0] - scenario["start"][0])
        if gazebo_mover:
            gazebo_mover.set_model_pose("start_marker", *scenario["start"], 0.005,
                                        world_name="level3_world")
            gazebo_mover.set_model_pose("goal_marker", *scenario["goal"], 0.005,
                                        world_name="level3_world")
            gazebo_mover.set_model_pose("autonomous_vehicle", *scenario["start"], 0.0,
                                        initial_yaw, world_name="level3_world")

        astar_result = execute(
            "AStar", astar, controller, scenario, None, args.max_time,
            astar_path, astar_latency, visualizer, gazebo_mover,
            level=3, trial_index=trial, dynamic_provider=pedestrian_positions,
        )
        if gazebo_mover:
            gazebo_mover.set_model_pose("autonomous_vehicle", *scenario["start"], 0.0,
                                        initial_yaw, world_name="level3_world")
        rrt_result = execute(
            "RRT", rrt, controller, scenario, rrt_seed, args.max_time,
            rrt_path, rrt_latency, visualizer, gazebo_mover,
            level=3, trial_index=trial, dynamic_provider=pedestrian_positions,
        )

        row = {"clutter": "dynamic", "trial": trial, "seed": args.seed,
               "start": scenario["start"], "goal": scenario["goal"],
               "obstacles": scenario["obstacles"], "pedestrian_count": 3}
        for prefix, result in (("astar", astar_result), ("rrt", rrt_result)):
            row.update({f"{prefix}_{key}": value for key, value in result.items()})
        rows.append(row)
        start_text = f"({scenario['start'][0]:.1f},{scenario['start'][1]:.1f})"
        goal_text = f"({scenario['goal'][0]:.1f},{scenario['goal'][1]:.1f})"
        astar_status = f"{astar_result['completion_time_s']:.1f}s" if astar_result["success"] else "FAIL"
        rrt_status = f"{rrt_result['completion_time_s']:.1f}s" if rrt_result["success"] else "FAIL"
        print(f"{trial:<4} {start_text:<16} {goal_text:<16} {3:<5} | "
              f"{astar_status:<9} {astar_result['path_length_m']:<8.2f} "
              f"{astar_result['planning_latency_ms']:<9.1f} | {rrt_status:<9} "
              f"{rrt_result['path_length_m']:<8.2f} {rrt_result['planning_latency_ms']:<9.1f}")
        wait_for_next_trial(trial, args.trials)

    write_rows(out / "level3_results.csv", rows)
    success_a = 100.0 * sum(row["astar_success"] for row in rows) / len(rows)
    success_r = 100.0 * sum(row["rrt_success"] for row in rows) / len(rows)
    print("=" * 112)
    print(" LEVEL 3 SUMMARY & STATISTICAL ANALYSIS")
    print("=" * 112)
    print(f"Goal Reach Success Rate: A*+DWB = {success_a:.1f}%, RRT+DWB = {success_r:.1f}%")
    with open(out / "level3_results.json", "w", encoding="utf-8") as stream:
        json.dump({"metadata": {"level": 3, "trials": args.trials, "seed": args.seed,
                                "dynamic_obstacles": 3},
                   "success_rates": {"astar": success_a, "rrt": success_r},
                   "trials": rows}, stream, indent=2)
    print(f"[OUTPUT] Trial metrics: {out / 'level3_results.csv'}, {out / 'level3_results.json'}")

    if visualizer:
        visualizer.node.destroy_node()
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
