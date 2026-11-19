import math
import random
import numpy as np

# RRT grows a tree of connected points to navigate around obstacles. Every point in that tree is represented
# by a node object.
def compute_urdf_footprint_radius(half_x=0.12, half_y=0.14, safety_margin=0.0):
    # Bounding radius covering the 0.24x0.24 m chassis and wheels at y = +- 0.14 m, plus optional buffer
    return math.hypot(half_x, half_y) + safety_margin

class Node:
    def __init__(self, x, y):
        # Tree node position in world meters
        self.x = x
        self.y = y
        # Pointer to parent node to be able to backtrack from goal to start once connected
        self.parent = None

class RRTPlanner:
    """Stores grid dimensions, metric resolution, world origin, and per-planner RNG"""
    def __init__(self, grid, resolution, origin_x, origin_y, robot_radius=None, safety_margin=0.0, seed=None):
        self.grid = np.array(grid)
        self.map_h, self.map_w = self.grid.shape
        self.res = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.safety_margin = float(safety_margin)
        self.rng = random.Random(seed)

        # Use explicit robot_radius if given
        # otherwise derive from URDF footprint
        initial_radius = compute_urdf_footprint_radius(0.12, 0.14, self.safety_margin) if robot_radius is None else float(robot_radius)
        self.set_robot_radius(initial_radius)

    def set_robot_radius(self, robot_radius):
        """Update active radius, convert to pixel inflation radius, and rebuild the blocked mask"""
        self.robot_radius = float(robot_radius)
        self.inflation_px = max(1, int(math.ceil(self.robot_radius / self.res))) 
        self.blocked = self._compute_inflated_grid()


    def _world_to_grid(self, x_m, y_m):
        """Convert metric (x, y) coordinates into (gx, gy) grid indices"""
        gx = int((x_m - self.origin_x) / self.res)
        gy = int((y_m - self.origin_y) / self.res)
        return gx, gy

    def _compute_inflated_grid(self):
        """
        Supports both 0-100 costmaps and the 0-255 PGM occupancy grids
        """
        if np.max(self.grid) <= 100 and np.any(self.grid == 100):
            is_obs = (self.grid >= 50) | (self.grid < 0)
        else:
            is_obs = self.grid < 128

        # Dilate each obstacle cell by a disk of radius inflation_px for point-robot collision checks
        blocked = np.zeros((self.map_h, self.map_w), dtype=bool)
        obs_y, obs_x = np.where(is_obs)
        r = self.inflation_px

        for oy, ox in zip(obs_y, obs_x):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = ox + dx, oy + dy
                    if 0 <= nx < self.map_w and 0 <= ny < self.map_h and dx * dx + dy <= r * r:
                        blocked[ny][nx] = True
        return blocked

    def _is_collision_free(self, x1, y1, x2, y2):
        """
        Sample points along the straight segment at half-pixel spacing and reject
        out-of-bounds or blocked cells
        """
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = int(dist / (self.res * 0.5)) + 1
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            gx, gy = self._world_to_grid(x, y)
            if not (0 <= gx < self.map_w and 0 <= gy < self.map_h) or self.blocked[gy][gx]:
                return False
        return True

    def plan(self, start_xy, goal_xy, max_iter=3000, step_size=0.4, goal_bias=0.1, seed=None):
        """
        Use per-call seed if supplied so repeated-seed studies stray reproducible
        """
        rng = random.Random(seed) if seed is not None else self.rng
        start_node = Node(start_xy[0], start_xy[1])
        goal_node = Node(goal_xy[0], goal_xy[1])
        node_list = [start_node]

        for _ in range(max_iter):
            # Sample the goal with probability goal_bias
            # otherwise sample uniformly across the map bounds
            if rng.random() < goal_bias:
                rnd_x, rnd_y = goal_node.x, goal_node.y
            else:
                rnd_x = rng.uniform(self.origin_x, self.origin_x + self.map_w * self.res)
                rnd_y = rng.uniform(self.origin_y, self.origin_y + self.map_h * self.res)

            # Find the closest existing node in the tree to the sampled point
            nearest_node = min(node_list, key=lambda n: math.hypot(n.x - rnd_x, n.y - rnd_y))

            # Advance at most step_size from the nearest_node toward the sample
            theta = math.atan2(rnd_y - nearest_node.y, rnd_x - nearest_node.x)
            dist = math.hypot(rnd_x - nearest_node.x, rnd_y - nearest_node.y)
            actual_step = min(step_size, dist)
            
            new_x = nearest_node.x + actual_step * math.cos(theta)
            new_y = nearest_node.y + actual_step * math.sin(theta)

            # Attach new_node if the edge is obstacle free
            if self._is_collision_free(nearest_node.x, nearest_node.y, new_x, new_y):
                new_node = Node(new_x, new_y)
                new_node.parent = nearest_node
                node_list.append(new_node)

                # Connect directly to goal_node if within step_size and line-of-sight is clear 
                if math.hypot(new_node.x - goal_node.x, new_node.y - goal_node.y) <= step_size:
                    if self._is_collision_free(new_node.x, new_node.y, goal_node.x, goal_node.y):
                        goal_node.parent = new_node
                        node_list.append(goal_node)
                        return self._extract_path(goal_node)

        return None 

    def _extract_path(self, node):
        """
        Walk parent links back from goal to start, then reverse for start-to-goal order
        """
        path = []
        current = node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]