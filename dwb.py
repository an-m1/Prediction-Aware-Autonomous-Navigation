import math
import numpy as np

# Nav2 DWB (Dynamic Window Approach) Local Controller
# Converts global reference path and obstacle map into optimal velocity commands (v, omega)
# under robot velocity and acceleration limits.

class DWBLocalPlanner:
    """
    Dynamic Window Approach (DWB) trajectory generator and evaluator.
    Follows differential-drive kinematics subject to velocity and acceleration limits.
    """

    def __init__(
        self,
        grid,
        resolution,
        origin_x,
        origin_y,
        robot_radius=0.18,
        max_vel_x=0.5,
        min_vel_x=0.0,
        max_vel_theta=1.0,
        min_vel_theta=-1.0,
        acc_lim_x=1.0,
        acc_lim_theta=2.0,
        decel_lim_x=-1.0,
        decel_lim_theta=-2.0,
        sim_time=1.0,
        sim_granularity=0.1,
        vx_samples=15,
        vtheta_samples=25,
        goal_tolerance=0.25,
        w_path=40.0,
        w_goal=4.0,
        w_align=14.0,
        w_speed=4.0,
        w_risk=0.0,
    ):
        self.grid = np.array(grid)
        self.map_h, self.map_w = self.grid.shape
        self.res = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.robot_radius = float(robot_radius)

        # Kinematic limits
        self.max_vel_x = float(max_vel_x)
        self.min_vel_x = float(min_vel_x)
        self.max_vel_theta = float(max_vel_theta)
        self.min_vel_theta = float(min_vel_theta)
        self.acc_lim_x = float(acc_lim_x)
        self.acc_lim_theta = float(acc_lim_theta)
        self.decel_lim_x = float(decel_lim_x)
        self.decel_lim_theta = float(decel_lim_theta)

        # Simulation lookahead
        self.sim_time = float(sim_time)
        self.sim_granularity = float(sim_granularity)
        self.sim_steps = max(2, int(round(self.sim_time / self.sim_granularity)))
        self.vx_samples = int(vx_samples)
        self.vtheta_samples = int(vtheta_samples)
        self.goal_tolerance = float(goal_tolerance)

        # Critic weights
        self.w_path = float(w_path)
        self.w_goal = float(w_goal)
        self.w_align = float(w_align)
        self.w_speed = float(w_speed)
        self.w_risk = float(w_risk)  # Level 3 prediction critic weight (0.0 = plain DWB)

        # Inflation mask for footprint collision checks
        self.inflation_px = int(math.ceil(self.robot_radius / self.res))
        self.blocked = self._compute_inflated_grid()

    def _world_to_grid(self, x_m, y_m):
        gx = int((x_m - self.origin_x) / self.res)
        gy = int((y_m - self.origin_y) / self.res)
        return gx, gy

    def _compute_inflated_grid(self):
        if np.max(self.grid) <= 100 and np.any(self.grid == 100):
            is_obs = (self.grid >= 50) | (self.grid == -1)
        else:
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

    def _compute_dynamic_window(self, current_v, current_w, dt):
        """Calculates dynamic window based on current velocity and acceleration limits."""
        v_min = max(self.min_vel_x, current_v + self.decel_lim_x * dt)
        v_max = min(self.max_vel_x, current_v + self.acc_lim_x * dt)

        w_min = max(self.min_vel_theta, current_w + self.decel_lim_theta * dt)
        w_max = min(self.max_vel_theta, current_w + self.acc_lim_theta * dt)

        return v_min, v_max, w_min, w_max

    def compute_velocity_commands(self, current_pose, current_vel, global_path, dt=0.1, last_waypoint_idx=0):
        """
        Vectorized computation of the optimal (v, omega) command to track global_path.
        current_pose: (x, y, theta)
        current_vel: (v, omega)
        global_path: list of (x, y) waypoints
        dt: controller time step
        last_waypoint_idx: monotonically increasing index along path to prevent backtracking
        """
        x, y, th = current_pose
        curr_v, curr_w = current_vel

        goal_x, goal_y = global_path[-1]
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)

        # Goal reached check
        if dist_to_goal <= self.goal_tolerance:
            return 0.0, 0.0, {"reached": True, "dist_to_goal": dist_to_goal, "waypoint_idx": len(global_path) - 1}

        # Calculate dynamic window
        v_min, v_max, w_min, w_max = self._compute_dynamic_window(curr_v, curr_w, dt)

        # Monotonic path waypoint tracking: search forward window around last_waypoint_idx
        search_start = max(0, last_waypoint_idx - 2)
        search_end = min(len(global_path), search_start + 40)
        sub_path = global_path[search_start:search_end]
        path_arr = np.array(sub_path)

        dists = np.hypot(path_arr[:, 0] - x, path_arr[:, 1] - y)
        sub_idx = int(np.argmin(dists))
        closest_idx = search_start + sub_idx

        # Lookahead: ~0.4m ahead along the path
        lookahead_idx = min(len(global_path) - 1, closest_idx + 4)
        target_pt = global_path[lookahead_idx]

        # Target heading: direction from current position toward lookahead waypoint
        target_heading = math.atan2(target_pt[1] - y, target_pt[0] - x)
        heading_err = abs(math.atan2(math.sin(target_heading - th), math.cos(target_heading - th)))

        # Heading governor: scale max forward speed if robot is pointed away from path
        effective_v_max = v_max * max(0.1, math.cos(min(heading_err, math.pi / 2)))
        v_cand = np.linspace(v_min, effective_v_max, self.vx_samples)
        w_cand = np.linspace(w_min, w_max, self.vtheta_samples)
        V_grid, W_grid = np.meshgrid(v_cand, w_cand)
        V = V_grid.flatten()
        W = W_grid.flatten()

        # High-performance vectorized trajectory rollout for all velocity candidates
        t_arr = np.arange(1, self.sim_steps + 1) * self.sim_granularity
        ths = th + W[:, None] * t_arr[None, :]
        xs = x + np.cumsum(V[:, None] * np.cos(ths) * self.sim_granularity, axis=1)
        ys = y + np.cumsum(V[:, None] * np.sin(ths) * self.sim_granularity, axis=1)

        # Collision check across all rollouts
        gxs = ((xs - self.origin_x) / self.res).astype(np.int32)
        gys = ((ys - self.origin_y) / self.res).astype(np.int32)

        out_of_bounds = (gxs < 0) | (gxs >= self.map_w) | (gys < 0) | (gys >= self.map_h)
        safe_gx = np.clip(gxs, 0, self.map_w - 1)
        safe_gy = np.clip(gys, 0, self.map_h - 1)
        collided = np.any(out_of_bounds | self.blocked[safe_gy, safe_gx], axis=1)

        valid_idx = np.where(~collided)[0]

        if len(valid_idx) == 0:
            # Fallback: rotate in place toward path target
            heading_err = math.atan2(target_pt[1] - y, target_pt[0] - x) - th
            heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))
            cmd_w = float(np.clip(heading_err * 1.5, self.min_vel_theta, self.max_vel_theta))
            return 0.0, cmd_w, {"reached": False, "dist_to_goal": dist_to_goal, "waypoint_idx": closest_idx}

        end_x = xs[valid_idx, -1]
        end_y = ys[valid_idx, -1]
        end_th = ths[valid_idx, -1]
        cand_v = V[valid_idx]

        # 1. PathDist Critic
        path_dist = np.hypot(target_pt[0] - end_x, target_pt[1] - end_y)

        # 2. GoalDist Critic
        goal_dist = np.hypot(goal_x - end_x, goal_y - end_y)

        # 3. PathAlign Critic: heading alignment to target waypoint direction
        angle_diff = np.abs(np.arctan2(np.sin(end_th - target_heading), np.cos(end_th - target_heading)))

        # 4. Speed Critic: forward progress reward
        speed_cost = (self.max_vel_x - cand_v) / max(self.max_vel_x, 1e-3)

        # Combined cost
        costs = (
            self.w_path * path_dist
            + self.w_goal * goal_dist
            + self.w_align * angle_diff
            + self.w_speed * speed_cost
        )

        best_i = valid_idx[np.argmin(costs)]
        best_v = float(V[best_i])
        best_w = float(W[best_i])

        return best_v, best_w, {"reached": False, "dist_to_goal": dist_to_goal, "waypoint_idx": closest_idx}
