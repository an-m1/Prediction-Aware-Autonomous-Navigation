import math
import argparse
import time
import numpy as np
import yaml
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from astar import AStarPlanner
from dwb import DWBLocalPlanner


class NavigationNode(Node):

    def __init__(self, args):
        super().__init__('navigation_node')

        self.args = args
        self.start = (args.start_x, args.start_y)
        self.goal = (args.goal_x, args.goal_y)

        self.pose = None
        self.velocity = (0.0, 0.0)
        self.global_path = None
        self.waypoint_idx = 0
        self.finished = False

        self.start_time = None
        self.planning_time = 0.0
        self.distance_travelled = 0.0
        self.previous_position = None

        self.actual_path = []

        map_dir = Path(__file__).resolve().parent
        map_file = map_dir / f'{args.map}_map.yaml'

        with open(map_file, 'r') as f:
            config = yaml.safe_load(f)

        image = Image.open(
            map_dir / config['image']
        ).convert('L')

        self.grid = np.flipud(np.array(image)).copy()

        self.resolution = config['resolution']
        self.origin_x = config['origin'][0]
        self.origin_y = config['origin'][1]

        if args.planner == 'astar':

            self.planner = AStarPlanner(
                self.grid,
                self.resolution,
                self.origin_x,
                self.origin_y,
                robot_radius=args.radius
            )

        elif args.planner == 'rrt':

            from rrt import RRTPlanner

            self.planner = RRTPlanner(
                self.grid,
                self.resolution,
                self.origin_x,
                self.origin_y,
                robot_radius=args.radius
            )

        self.dwb = DWBLocalPlanner(
            self.grid,
            self.resolution,
            self.origin_x,
            self.origin_y,
            robot_radius=args.radius,
            max_vel_x=args.speed,
            w_risk=0.0
        )

        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.timer = self.create_timer(
            0.1, self.control_loop
        )

        self.get_logger().info(f'Map: {args.map}')
        self.get_logger().info(f'Planner: {args.planner.upper()}')
        self.get_logger().info('Local controller: DWB')

    def odom_callback(self, msg):

        position = msg.pose.pose.position
        q = msg.pose.pose.orientation

        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        x = position.x + self.start[0]
        y = position.y + self.start[1]

        self.pose = (x, y, theta)

        self.velocity = (
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        )

        if self.start_time is not None and not self.finished:

            self.actual_path.append((x, y))

            if self.previous_position is not None:

                self.distance_travelled += math.hypot(
                    x - self.previous_position[0],
                    y - self.previous_position[1]
                )

            self.previous_position = (x, y)

    def stop_robot(self):

        self.cmd_pub.publish(Twist())

    def plot_results(self):

        height, width = self.grid.shape

        extent = [
            self.origin_x,
            self.origin_x + width * self.resolution,
            self.origin_y,
            self.origin_y + height * self.resolution
        ]

        plt.figure(figsize=(8, 8))

        plt.imshow(
            self.grid,
            cmap='gray',
            origin='lower',
            extent=extent,
            vmin=0,
            vmax=255
        )

        planned = np.array(self.global_path)
        actual = np.array(self.actual_path)

        plt.plot(
            planned[:, 0],
            planned[:, 1],
            color='royalblue',
            linewidth=2,
            label=f'{self.args.planner.upper()} Planned Path'
        )

        if len(actual) > 0:

            plt.plot(
                actual[:, 0],
                actual[:, 1],
                color='darkorange',
                linewidth=2,
                linestyle='--',
                label='Actual Robot Trajectory'
            )

        plt.scatter(
            self.start[0],
            self.start[1],
            color='green',
            s=90,
            label='Start',
            zorder=5
        )

        plt.scatter(
            self.goal[0],
            self.goal[1],
            color='red',
            s=90,
            label='Goal',
            zorder=5
        )

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')

        plt.title(
            f'{self.args.map.upper()} - '
            f'{self.args.planner.upper()} + DWB'
        )

        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])

        plt.axis('equal')
        plt.grid(alpha=0.2)
        plt.legend(loc='upper right')

        plt.tight_layout()

        results_dir = Path(__file__).resolve().parent / 'results'
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        filename = results_dir / (
            f'{self.args.map}_{self.args.planner}_{timestamp}.png'
        )

        plt.savefig(filename, dpi=300, bbox_inches='tight')

        print(f'Graph saved to: {filename}')

        plt.show()

    def control_loop(self):

        if self.finished:
            self.stop_robot()
            return

        if self.pose is None:
            return

        if self.global_path is None:

            planning_start = time.perf_counter()

            self.global_path = self.planner.plan(
                self.pose[:2],
                self.goal
            )

            self.planning_time = (
                time.perf_counter() - planning_start
            )

            if self.global_path is None or len(self.global_path) == 0:

                self.get_logger().error('No path found')
                self.finished = True
                self.stop_robot()
                return

            self.global_path = AStarPlanner.resample_path(
                list(self.global_path),
                spacing=0.1
            )

            self.path_length = sum(
                math.hypot(
                    p2[0] - p1[0],
                    p2[1] - p1[1]
                )
                for p1, p2 in zip(
                    self.global_path[:-1],
                    self.global_path[1:]
                )
            )

            self.get_logger().info(
                f'Waypoints: {len(self.global_path)}'
            )

            self.get_logger().info(
                f'Planning time: {self.planning_time:.4f} s'
            )

            self.get_logger().info(
                f'Planned path length: {self.path_length:.3f} m'
            )

            self.start_time = time.perf_counter()
            self.previous_position = self.pose[:2]
            self.actual_path.append(self.pose[:2])

        v, w, info = self.dwb.compute_velocity_commands(
            current_pose=self.pose,
            current_vel=self.velocity,
            global_path=self.global_path,
            dt=0.1,
            last_waypoint_idx=self.waypoint_idx
        )

        self.waypoint_idx = max(
            self.waypoint_idx,
            info['waypoint_idx']
        )

        if info['reached']:

            self.stop_robot()
            self.finished = True

            navigation_time = (
                time.perf_counter() - self.start_time
            )

            self.get_logger().info(
                'GOAL REACHED SUCCESSFULLY!'
            )

            print('\n========== RESULTS ==========')
            print(f'Planning time: {self.planning_time:.4f} s')
            print(f'Navigation time: {navigation_time:.2f} s')
            print(f'Planned path length: {self.path_length:.3f} m')
            print(f'Odometry distance: {self.distance_travelled:.3f} m')
            print('Success: YES')
            print('=============================\n')

            return

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)

        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'Position: ({self.pose[0]:.2f}, '
            f'{self.pose[1]:.2f}) | '
            f'Goal distance: {info["dist_to_goal"]:.2f} m | '
            f'v: {v:.2f} | w: {w:.2f}',
            throttle_duration_sec=1.0
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--map',
        choices=['level1', 'level2', 'level3'],
        default='level1'
    )

    parser.add_argument(
        '--planner',
        choices=['astar', 'rrt'],
        default='astar'
    )

    parser.add_argument('--start-x', type=float, default=-4.0)
    parser.add_argument('--start-y', type=float, default=0.0)

    parser.add_argument('--goal-x', type=float, default=4.0)
    parser.add_argument('--goal-y', type=float, default=0.0)

    parser.add_argument('--speed', type=float, default=0.2)
    parser.add_argument('--radius', type=float, default=0.35)

    args = parser.parse_args()

    rclpy.init(args=[])

    node = NavigationNode(args)

    try:

        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)

        if node.finished and node.global_path and node.actual_path:

            node.stop_robot()

            node.plot_results()

    except KeyboardInterrupt:

        pass

    finally:

        if rclpy.ok():
            node.stop_robot()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
