#!/usr/bin/env python3
"""Shared live top-down visualizer for navigation Levels 1, 2, and 3."""

import argparse
import json
import math
import tkinter as tk

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class NavigationVisualizer(Node):
    """Display planner searches, paths, robot motion, and dynamic obstacles."""

    CANVAS_SIZE = 820
    MARGIN = 42

    def __init__(self, level):
        self.level = int(level)
        self.topic_prefix = f"/level{self.level}/visualization"
        super().__init__(f"level{self.level}_topdown_visualizer")

        scenario_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, f"{self.topic_prefix}/scenario", self._scenario_callback,
            scenario_qos,
        )
        self.create_subscription(
            PoseStamped, f"{self.topic_prefix}/robot_pose", self._pose_callback, 20,
        )
        self.create_subscription(
            String, f"{self.topic_prefix}/pedestrians", self._pedestrians_callback, 20,
        )
        self.create_subscription(Odometry, "/odom", self._odom_callback, 20)

        self.scenario = None
        self.robot_pose = None
        self.have_visualization_pose = False
        self.trail = []
        self.pedestrians = []

        self.root = tk.Tk()
        self.root.title(f"Level {self.level} Planner Map: A* and RRT")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.canvas = tk.Canvas(
            self.root, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            background="white", highlightthickness=0,
        )
        self.canvas.pack()
        self._draw_waiting()
        self.root.after(20, self._spin)

    def _close(self):
        self.root.destroy()
        if rclpy.ok():
            rclpy.shutdown()

    def _spin(self):
        if rclpy.ok() and self.root.winfo_exists():
            rclpy.spin_once(self, timeout_sec=0.0)
            self.root.after(20, self._spin)

    def _scenario_callback(self, message):
        try:
            scenario = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Ignoring invalid scenario message: {exc}")
            return

        # Level 1 historically publishes one `obstacle` with different field
        # names. Normalize it to the list schema used by Levels 2 and 3.
        if "obstacles" not in scenario and "obstacle" in scenario:
            obstacle = scenario["obstacle"]
            scenario["obstacles"] = [{
                "x": obstacle["cx"], "y": obstacle["cy"],
                "w": obstacle["width"], "h": obstacle["height"],
                "yaw": obstacle.get("yaw", 0.0),
            }]

        self.scenario = scenario
        self.trail = []
        self.pedestrians = []
        self.robot_pose = None
        self.have_visualization_pose = False
        self._draw_static_map()

    def _pose_callback(self, message):
        q = message.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.have_visualization_pose = True
        self._update_robot(message.pose.position.x, message.pose.position.y, yaw)

    def _odom_callback(self, message):
        if self.have_visualization_pose:
            return
        q = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self._update_robot(message.pose.pose.position.x,
                           message.pose.pose.position.y, yaw)

    def _pedestrians_callback(self, message):
        try:
            self.pedestrians = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Ignoring invalid pedestrian message: {exc}")
            return
        self._draw_pedestrians()

    def _world_to_canvas(self, x, y):
        half = self.scenario.get("arena_half", 6.0) if self.scenario else 6.0
        side = self.CANVAS_SIZE - 2 * self.MARGIN
        px = self.MARGIN + (x + half) * side / (2.0 * half)
        py = self.CANVAS_SIZE - self.MARGIN - (y + half) * side / (2.0 * half)
        return px, py

    def _draw_waiting(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.CANVAS_SIZE / 2, self.CANVAS_SIZE / 2,
            text=f"Waiting for Level {self.level} trial data...",
            fill="#202020", font=("TkDefaultFont", 15),
        )

    def _draw_static_map(self):
        self.canvas.delete("all")
        if not self.scenario:
            self._draw_waiting()
            return

        half = self.scenario.get("arena_half", 6.0)
        left, top = self._world_to_canvas(-half, half)
        right, bottom = self._world_to_canvas(half, -half)
        self.canvas.create_rectangle(left, top, right, bottom,
                                     outline="#202020", width=3)

        for edges in (self.scenario.get("astar_edges", []),
                      self.scenario.get("rrt_edges", [])):
            for (ax, ay), (bx, by) in edges:
                x1, y1 = self._world_to_canvas(ax, ay)
                x2, y2 = self._world_to_canvas(bx, by)
                self.canvas.create_line(x1, y1, x2, y2,
                                        fill="#b5b5b5", width=1)

        for obstacle in self.scenario.get("obstacles", []):
            self._draw_obstacle(obstacle)

        self._draw_path(self.scenario.get("astar_path", []), "#1769e0", 4)
        self._draw_path(self.scenario.get("rrt_path", []), "#d12f2f", 4)
        self._draw_marker(self.scenario["start"], "#20a34a", "Start")
        self._draw_marker(self.scenario["goal"], "#e04444", "Goal")

        clutter = self.scenario.get("clutter")
        subtitle = f" - {clutter}" if clutter else ""
        self.canvas.create_text(
            self.CANVAS_SIZE / 2, 18,
            text=f"Level {self.level}{subtitle} - Trial {self.scenario['trial']} - planner map",
            fill="#202020", font=("TkDefaultFont", 13, "bold"),
        )
        self._draw_legend()
        self._draw_pedestrians()

    def _draw_obstacle(self, obstacle):
        cx, cy = obstacle["x"], obstacle["y"]
        width, height = obstacle["w"], obstacle["h"]
        yaw = obstacle.get("yaw", 0.0)
        corners = []
        for lx, ly in ((-width / 2, -height / 2),
                       (width / 2, -height / 2),
                       (width / 2, height / 2),
                       (-width / 2, height / 2)):
            wx = cx + math.cos(yaw) * lx - math.sin(yaw) * ly
            wy = cy + math.sin(yaw) * lx + math.cos(yaw) * ly
            corners.extend(self._world_to_canvas(wx, wy))
        self.canvas.create_polygon(corners, fill="#666666",
                                   outline="#202020", width=2)

    def _draw_path(self, path, color, width):
        if len(path) < 2:
            return
        points = []
        for x, y in path:
            points.extend(self._world_to_canvas(x, y))
        self.canvas.create_line(*points, fill=color, width=width,
                                joinstyle=tk.ROUND)

    def _draw_marker(self, point, color, label):
        x, y = self._world_to_canvas(*point)
        radius = 8
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                fill=color, outline="#202020", width=1)
        self.canvas.create_text(x, y - 16, text=label, fill="#202020",
                                font=("TkDefaultFont", 10, "bold"))

    def _draw_legend(self):
        entries = [
            ("#666666", "Obstacle"), ("#20a34a", "Start"),
            ("#e04444", "Goal"), ("#b5b5b5", "Explored branches"),
            ("#1769e0", "A* final path"), ("#d12f2f", "RRT final path"),
            ("#7b3fc6", "Live robot / trail"),
        ]
        if self.level == 3:
            entries.append(("#f09a32", "Moving pedestrian"))
        x, y = 52, 45
        box_bottom = 52 + 19 * len(entries)
        self.canvas.create_rectangle(42, 30, 226, box_bottom,
                                     fill="white", outline="#202020")
        self.canvas.create_text(52, 42, anchor="w", text="Legend",
                                fill="#202020", font=("TkDefaultFont", 10, "bold"))
        for color, text in entries:
            y += 19
            self.canvas.create_line(x, y, x + 22, y, fill=color, width=4)
            self.canvas.create_text(x + 30, y, anchor="w", text=text,
                                    fill="#202020", font=("TkDefaultFont", 9))

    def _draw_pedestrians(self):
        self.canvas.delete("dynamic")
        if not self.scenario:
            return
        for pedestrian in self.pedestrians:
            x, y = self._world_to_canvas(pedestrian["x"], pedestrian["y"])
            radius = 10
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                    fill="#f09a32", outline="#7a3e00", width=2,
                                    tags="dynamic")

    def _update_robot(self, x, y, yaw):
        if not self.scenario:
            return
        self.robot_pose = (x, y, yaw)
        self.trail.append((x, y))
        self.trail = self.trail[-1500:]
        self.canvas.delete("live")

        if len(self.trail) > 1:
            points = []
            for trail_x, trail_y in self.trail:
                points.extend(self._world_to_canvas(trail_x, trail_y))
            self.canvas.create_line(*points, fill="#b98ad8", width=2, tags="live")

        px, py = self._world_to_canvas(x, y)
        radius = 9
        self.canvas.create_oval(px - radius, py - radius, px + radius, py + radius,
                                fill="#7b3fc6", outline="#202020", width=2,
                                tags="live")
        tip_x = px + 20 * math.cos(yaw)
        tip_y = py - 20 * math.sin(yaw)
        self.canvas.create_line(px, py, tip_x, tip_y, fill="white", width=3,
                                arrow=tk.LAST, tags="live")
        self.canvas.create_text(px, py + 18, text="Robot", fill="#202020",
                                font=("TkDefaultFont", 9, "bold"), tags="live")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    arguments, ros_arguments = parser.parse_known_args()
    rclpy.init(args=ros_arguments)
    node = NavigationVisualizer(arguments.level)
    try:
        node.root.mainloop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
