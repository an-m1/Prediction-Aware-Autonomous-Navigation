import heapq
import math
import numpy as np

# A* (A-Star) is a grid-based, deterministic global path planning algorithm.
# It uses an admissible heuristic (Euclidean distance) to guarantee an optimal path
# on the discrete occupancy grid.

class AStarPlanner:
    """Sets up the A* planner with an occupancy grid, metric resolution (0.05 m/px),
    origin coordinates, and the robot's physical radius for obstacle inflation."""

    def __init__(self, grid, resolution, origin_x, origin_y, robot_radius=0.18):
        self.grid = np.array(grid)
        self.map_h, self.map_w = self.grid.shape
        self.res = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.robot_radius = float(robot_radius)
        # Converts robot's physical radius to pixels for safety boundary inflation
        self.inflation_px = int(math.ceil(self.robot_radius / self.res))
        # Pre-computes the obstacle mask so path planning queries stay fast
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
        Matches the exact costmap inflation used by RRT for fair comparison.
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

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    ny = np.clip(obs_y + dy, 0, self.map_h - 1)
                    nx = np.clip(obs_x + dx, 0, self.map_w - 1)
                    blocked[ny, nx] = True

        return blocked

    def _is_point_valid(self, x, y):
        """Checks if continuous world coordinate is inside the map and clear of obstacles."""
        gx, gy = self._world_to_grid(x, y)
        if not (0 <= gx < self.map_w and 0 <= gy < self.map_h):
            return False
        return not self.blocked[gy, nx] if False else not self.blocked[gy, gx]

    def plan(self, start_xy, goal_xy):
        """
        Plans an optimal path from start_xy to goal_xy on the 8-connected grid.
        Returns a list of (x, y) tuples from start to goal in meters, or None if no path found.
        """
        start_x, start_y = float(start_xy[0]), float(start_xy[1])
        goal_x, goal_y = float(goal_xy[0]), float(goal_xy[1])
        self.last_explored_edges = []

        start_gx, start_gy = self._world_to_grid(start_x, start_y)
        goal_gx, goal_gy = self._world_to_grid(goal_x, goal_y)

        # Validate start and goal positions
        if not (0 <= start_gx < self.map_w and 0 <= start_gy < self.map_h) or self.blocked[start_gy, start_gx]:
            return None
        if not (0 <= goal_gx < self.map_w and 0 <= goal_gy < self.map_h) or self.blocked[goal_gy, goal_gx]:
            return None

        if (start_gx, start_gy) == (goal_gx, goal_gy):
            return [(start_x, start_y), (goal_x, goal_y)]

        # 8-connected grid moves with (dx, dy, move_cost)
        SQRT2 = math.sqrt(2.0)
        motions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (-1, -1, SQRT2)
        ]

        # Priority queue stores: (f_score, counter, (gx, gy))
        counter = 0
        open_set = []
        start_h = math.hypot(start_gx - goal_gx, start_gy - goal_gy) * self.res
        heapq.heappush(open_set, (start_h, counter, (start_gx, start_gy)))

        g_scores = {(start_gx, start_gy): 0.0}
        came_from = {}
        closed_set = set()

        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            cgx, cgy = current

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == (goal_gx, goal_gy):
                # Goal reached: reconstruct path
                raw_path = [current]
                while raw_path[-1] in came_from:
                    raw_path.append(came_from[raw_path[-1]])
                raw_path.reverse()

                # Convert to world meters and attach exact start and goal coordinates
                world_path = [(start_x, start_y)]
                for gx, gy in raw_path[1:-1]:
                    world_path.append(self._grid_to_world(gx, gy))
                world_path.append((goal_x, goal_y))
                return world_path

            current_g = g_scores[current]

            for dx, dy, move_cost in motions:
                ngx, ngy = cgx + dx, cgy + dy
                neighbor = (ngx, ngy)

                if neighbor in closed_set:
                    continue

                if not (0 <= ngx < self.map_w and 0 <= ngy < self.map_h):
                    continue

                if self.blocked[ngy, ngx]:
                    continue

                # Prevent diagonal corner cutting through adjacent obstacles
                if dx != 0 and dy != 0:
                    if self.blocked[cgy, ngx] and self.blocked[ngy, cgx]:
                        continue

                tentative_g = current_g + move_cost * self.res

                if tentative_g < g_scores.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    if len(self.last_explored_edges) < 4000:
                        self.last_explored_edges.append((
                            self._grid_to_world(cgx, cgy),
                            self._grid_to_world(ngx, ngy),
                        ))
                    h_score = math.hypot(ngx - goal_gx, ngy - goal_gy) * self.res
                    f_score = tentative_g + h_score
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return None  # Path not found

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
