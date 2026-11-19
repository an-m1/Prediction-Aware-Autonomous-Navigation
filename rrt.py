import math
import random
import numpy as np

# RRT grows a tree of connected points to navigate around obstacles.
# Every point in that tree is represented by a Node object.

class Node:
    def __init__(self, x, y):
        # Stores (x, y) in meters
        self.x = float(x)
        self.y = float(y)
        # Pointer back to parent node for path reconstruction
        self.parent = None
        self.cost = 0.0


class RRTPlanner:
    """Sets up the planner by loading the occupancy grid map, dimensions, metric resolution (0.05 m/px),
    map origin, and the robot's physical radius (0.18 m being used as a safety buffer)."""

    def __init__(self, grid, resolution, origin_x, origin_y, robot_radius=0.18):
        self.grid = np.array(grid)
        self.map_h, self.map_w = self.grid.shape
        self.res = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.robot_radius = float(robot_radius)
        # Converts robot's physical radius to pixels to know buffer space away from walls and obstacles
        self.inflation_px = int(math.ceil(self.robot_radius / self.res))
        # Pre-calculates an obstacle mask ahead of time so planner doesn't recalculate safety boundaries every iteration
        self.blocked = self._compute_inflated_grid()
        self.last_explored_edges = []

    def _world_to_grid(self, x_m, y_m):
        gx = int((x_m - self.origin_x) / self.res)
        gy = int((y_m - self.origin_y) / self.res)
        return gx, gy

    def _grid_to_world(self, gx, gy):
        x_m = self.origin_x + (gx + 0.5) * self.res
        y_m = self.origin_y + (gy + 0.5) * self.res
        return x_m, y_m

    def _compute_inflated_grid(self):
        """
        Finds every obstacle pixel and puffs it outward by a circle of radius self.inflation_px.
        Treats the robot as a single point moving on an inflated map, ensuring the robot's physical
        edges won't scrape against walls or obstacles.
        Supports both raw PGM image format (0 = occupied) and ROS OccupancyGrid format (100 = occupied).
        """
        if np.max(self.grid) <= 100 and np.any(self.grid == 100):
            # ROS OccupancyGrid convention: 100 is occupied, -1 is unknown
            is_obs = (self.grid >= 50) | (self.grid == -1)
        else:
            # Image / PGM convention: 0 is occupied (black), 255 is free (white)
            is_obs = (self.grid < 128)

        r = self.inflation_px
        blocked = np.copy(is_obs)
        obs_y, obs_x = np.where(is_obs)

        # Fast vectorized circular dilation over all obstacle pixels
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    ny = np.clip(obs_y + dy, 0, self.map_h - 1)
                    nx = np.clip(obs_x + dx, 0, self.map_w - 1)
                    blocked[ny, nx] = True

        return blocked

    def _is_point_valid(self, x, y):
        """Checks if a continuous world coordinate lies inside map boundaries and outside inflated obstacles."""
        gx, gy = self._world_to_grid(x, y)
        if not (0 <= gx < self.map_w and 0 <= gy < self.map_h):
            return False
        return not self.blocked[gy, gx]

    def _is_collision_free(self, x1, y1, x2, y2):
        """
        Checks whether a straight line path drawn between two points (x1, y1 and x2, y2) passes through any blocked/inflated pixels.
        Chops the line into tiny segments and tests each point against self.blocked. If even one point hits an obstacle, returns False.
        """
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1e-6:
            return self._is_point_valid(x1, y1)

        steps = int(math.ceil(dist / (self.res * 0.5))) + 1
        ts = np.linspace(0.0, 1.0, steps)
        xs = x1 + ts * (x2 - x1)
        ys = y1 + ts * (y2 - y1)
        gxs = ((xs - self.origin_x) / self.res).astype(np.int32)
        gys = ((ys - self.origin_y) / self.res).astype(np.int32)

        if np.any((gxs < 0) | (gxs >= self.map_w) | (gys < 0) | (gys >= self.map_h)):
            return False
        return not np.any(self.blocked[gys, gxs])

    def plan(self, start_xy, goal_xy, max_iter=3000, step_size=0.4, goal_bias=0.1, seed=None):
        """
        Plans a path from start_xy to goal_xy using RRT with goal biasing.
        Returns a list of (x, y) tuples from start to goal in meters, or None if no path found within max_iter.
        seed: optional random seed for reproducible evaluation trials.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.last_explored_edges = []

        start_x, start_y = float(start_xy[0]), float(start_xy[1])
        goal_x, goal_y = float(goal_xy[0]), float(goal_xy[1])

        if not self._is_point_valid(start_x, start_y):
            return None
        if not self._is_point_valid(goal_x, goal_y):
            return None

        # Direct line-of-sight check
        if self._is_collision_free(start_x, start_y, goal_x, goal_y):
            return [(start_x, start_y), (goal_x, goal_y)]

        start_node = Node(start_x, start_y)
        goal_node = Node(goal_x, goal_y)
        node_list = [start_node]

        # Fast coordinate tracking arrays for vectorized nearest-neighbor search
        node_x = [start_x]
        node_y = [start_y]

        x_min, x_max = self.origin_x, self.origin_x + self.map_w * self.res
        y_min, y_max = self.origin_y, self.origin_y + self.map_h * self.res

        for _ in range(max_iter):
            # 1. Goal bias sampling
            if random.random() < goal_bias:
                rnd_x, rnd_y = goal_node.x, goal_node.y
            else:
                rnd_x = random.uniform(x_min, x_max)
                rnd_y = random.uniform(y_min, y_max)

            # 2. Nearest neighbor search using vectorized distance computation
            arr_x = np.array(node_x)
            arr_y = np.array(node_y)
            dists_sq = (arr_x - rnd_x) ** 2 + (arr_y - rnd_y) ** 2
            nearest_idx = int(np.argmin(dists_sq))
            nearest_node = node_list[nearest_idx]

            # 3. Steering towards sampled point by step_size
            dist = math.sqrt(dists_sq[nearest_idx])
            if dist < 1e-6:
                continue

            actual_step = min(step_size, dist)
            theta = math.atan2(rnd_y - nearest_node.y, rnd_x - nearest_node.x)
            new_x = nearest_node.x + actual_step * math.cos(theta)
            new_y = nearest_node.y + actual_step * math.sin(theta)

            # 4. Collision check & add to tree
            if self._is_collision_free(nearest_node.x, nearest_node.y, new_x, new_y):
                new_node = Node(new_x, new_y)
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + actual_step
                node_list.append(new_node)
                node_x.append(new_x)
                node_y.append(new_y)
                self.last_explored_edges.append(
                    ((nearest_node.x, nearest_node.y), (new_x, new_y))
                )

                # 5. Goal check: can we reach goal directly from new node?
                dist_to_goal = math.hypot(new_x - goal_node.x, new_y - goal_node.y)
                if dist_to_goal <= step_size:
                    if self._is_collision_free(new_x, new_y, goal_node.x, goal_node.y):
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + dist_to_goal
                        return self._shortcut_path(self._extract_path(goal_node))

        return None  # Path not found within max_iter


    def _shortcut_path(self, path):
        """Greedily remove unnecessary RRT bends while preserving collision safety."""
        if len(path) < 3:
            return path
        result = [path[0]]
        current = 0
        while current < len(path) - 1:
            furthest = len(path) - 1
            while furthest > current + 1 and not self._is_collision_free(
                    *path[current], *path[furthest]):
                furthest -= 1
            result.append(path[furthest])
            current = furthest
        return result

    def _extract_path(self, node):
        """Starting from goal node, traces parents back to start and reverses."""
        path = []
        curr = node
        while curr is not None:
            path.append((curr.x, curr.y))
            curr = curr.parent
        return path[::-1]

    @staticmethod
    def resample_path(path, spacing=0.1):
        """Resamples path into equidistant waypoints for smooth local trajectory tracking."""
        if not path or len(path) < 2:
            return path
        resampled = [path[0]]
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist < 1e-6:
                continue
            steps = max(1, int(math.ceil(dist / spacing)))
            for s in range(1, steps):
                t = s / float(steps)
                resampled.append((p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])))
            resampled.append(p2)
        return resampled
