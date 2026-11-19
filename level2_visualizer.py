#!/usr/bin/env python3
"""Live top-down map for Level 2 A* / RRT + DWB trials."""

import json
import math
import tkinter as tk

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class Level1Visualizer(Node):
    """ROS 2 subscriber and Tk canvas for planner overlays and live tracking."""

    CANVAS_SIZE = 820
    MARGIN = 42
    LEVEL = 2
    TOPIC_PREFIX = "/level2/visualization"

    def __init__(self):
        super().__init__(f"level{self.LEVEL}_topdown_visualizer")
        scenario_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, f"{self.TOPIC_PREFIX}/scenario", self._scenario_callback, scenario_qos
        )
        self.create_subscription(
            PoseStamped, f"{self.TOPIC_PREFIX}/robot_pose", self._pose_callback, 20
        )
        self.create_subscription(
            String, f"{self.TOPIC_PREFIX}/pedestrians", self._pedestrians_callback, 20
        )
        # Subscribe to odometry as a fallback for deployments that publish a
        # global /odom pose rather than the Level 2 visualization pose.
        self.create_subscription(Odometry, "/odom", self._odom_callback, 20)

        self.scenario = None
        self.robot_pose = None
        self.have_visualization_pose = False
        self.trail = []
        self.pedestrians = []

        self.root = tk.Tk()
        self.root.title(f"Level {self.LEVEL} Planner Map: A* and RRT")
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
        rclpy.shutdown()

    def _spin(self):
        if rclpy.ok() and self.root.winfo_exists():
            rclpy.spin_once(self, timeout_sec=0.0)
            self.root.after(20, self._spin)

    def _scenario_callback(self, message):
        try:
            self.scenario = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Ignoring invalid scenario message: {exc}")
            return
        self.trail = []
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

    def _pedestrians_callback(self, message):
        try:
            self.pedestrians = json.loads(message.data)
        except json.JSONDecodeError:
            return
        self.canvas.delete("dynamic")
        for pedestrian in self.pedestrians:
            x, y = self._world_to_canvas(pedestrian["x"], pedestrian["y"])
            radius = 10
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                    fill="#f09a32", outline="#7a3e00", width=2,
                                    tags="dynamic")

    def _odom_callback(self, message):
        if self.have_visualization_pose:
            return
        q = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.x * q.z),
        )
        self._update_robot(message.pose.pose.position.x, message.pose.pose.position.y, yaw)

    def _world_to_canvas(self, x, y):
        half = self.scenario["arena_half"] if self.scenario else 6.0
        side = self.CANVAS_SIZE - 2 * self.MARGIN
        px = self.MARGIN + (x + half) * side / (2.0 * half)
        py = self.CANVAS_SIZE - self.MARGIN - (y + half) * side / (2.0 * half)
        return px, py

    def _draw_waiting(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.CANVAS_SIZE / 2, self.CANVAS_SIZE / 2,
            text=f"Waiting for Level {self.LEVEL} trial data...", fill="#202020",
            font=("TkDefaultFont", 15),
        )

    def _draw_static_map(self):
        self.canvas.delete("all")
        if not self.scenario:
            self._draw_waiting()
            return

        half = self.scenario["arena_half"]
        left, top = self._world_to_canvas(-half, half)
        right, bottom = self._world_to_canvas(half, -half)
        self.canvas.create_rectangle(left, top, right, bottom, outline="#202020", width=3)

        # All visited A* graph edges and RRT tree branches are visible first.
        for edges in (self.scenario.get("astar_edges", []), self.scenario.get("rrt_edges", [])):
            for (ax, ay), (bx, by) in edges:
                x1, y1 = self._world_to_canvas(ax, ay)
                x2, y2 = self._world_to_canvas(bx, by)
                self.canvas.create_line(x1, y1, x2, y2, fill="#b5b5b5", width=1)

        for obstacle in self.scenario.get("obstacles", []):
            cx, cy = obstacle["x"], obstacle["y"]
            width, height, yaw = obstacle["w"], obstacle["h"], obstacle["yaw"]
            corners = []
            for lx, ly in ((-width / 2, -height / 2), (width / 2, -height / 2),
                           (width / 2, height / 2), (-width / 2, height / 2)):
                wx = cx + math.cos(yaw) * lx - math.sin(yaw) * ly
                wy = cy + math.sin(yaw) * lx + math.cos(yaw) * ly
                corners.extend(self._world_to_canvas(wx, wy))
            self.canvas.create_polygon(corners, fill="#666666", outline="#202020", width=2)

        self._draw_path(self.scenario.get("astar_path", []), "#1769e0", 4)
        self._draw_path(self.scenario.get("rrt_path", []), "#d12f2f", 4)
        self._draw_marker(self.scenario["start"], "#20a34a", "Start")
        self._draw_marker(self.scenario["goal"], "#e04444", "Goal")

        self.canvas.create_text(
            self.CANVAS_SIZE / 2, 18,
            text=f"Level {self.LEVEL} - Trial {self.scenario['trial']} - top-down planner map",
            fill="#202020", font=("TkDefaultFont", 13, "bold"),
        )
        self._draw_legend()

    def _draw_path(self, path, color, width):
        if len(path) < 2:
            return
        points = []
        for x, y in path:
            points.extend(self._world_to_canvas(x, y))
        self.canvas.create_line(*points, fill=color, width=width, joinstyle=tk.ROUND)

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
            ("#f09a32", "Moving pedestrian"),
        ]
        x, y = 52, 45
        self.canvas.create_rectangle(42, 30, 226, 184, fill="white", outline="#202020")
        self.canvas.create_text(52, 42, anchor="w", text="Legend", fill="#202020",
                                font=("TkDefaultFont", 10, "bold"))
        for color, text in entries:
            y += 19
            self.canvas.create_line(x, y, x + 22, y, fill=color, width=4)
            self.canvas.create_text(x + 30, y, anchor="w", text=text, fill="#202020",
                                    font=("TkDefaultFont", 9))

    def _update_robot(self, x, y, yaw):
        if not self.scenario:
            return
        self.robot_pose = (x, y, yaw)
        self.trail.append((x, y))
        self.trail = self.trail[-1500:]
        self.canvas.delete("live")

        if len(self.trail) > 1:
            points = []
            for tx, ty in self.trail:
                points.extend(self._world_to_canvas(tx, ty))
            self.canvas.create_line(*points, fill="#b98ad8", width=2, tags="live")

        px, py = self._world_to_canvas(x, y)
        radius = 9
        self.canvas.create_oval(px - radius, py - radius, px + radius, py + radius,
                                fill="#7b3fc6", outline="#202020", width=2, tags="live")
        arrow_length = 20
        tip_x = px + arrow_length * math.cos(yaw)
        tip_y = py - arrow_length * math.sin(yaw)
        self.canvas.create_line(px, py, tip_x, tip_y, fill="white", width=3,
                                arrow=tk.LAST, tags="live")
        self.canvas.create_text(px, py + 18, text="Robot", fill="#202020",
                                font=("TkDefaultFont", 9, "bold"), tags="live")


def main():
    rclpy.init()
    node = Level1Visualizer()
    try:
        node.root.mainloop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
