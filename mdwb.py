import math
import numpy as np

from dwb import DWBLocalPlanner

# Discrete motion primitives: 8 compass directions + 1 stop state
MOTION_MODES = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0), (0.0, 0.0),]

# Unit direction vectors for each motion mode
_MOTION_DIRS = np.array([[dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)] if (dx or dy) else [0.0, 0.0] for dx, dy in MOTION_MODES], dtype=float,)


def build_default_transition_matrix(p_continue=0.70, p_turn_each=0.10, p_stop=0.10):
    """Builds a 9x9 transition matric T for continuing, turning, or stopping."""
    n_modes = len(MOTION_MODES)
    move_indices = list(range(8))
    stop_idx = 8
    T = np.zeros((n_modes, n_modes), dtype=float)

    # Transtion probabilities from active movement modes
    for m in move_indices:
        left_idx = move_indices[(m + 1) % 8]
        right_idx = move_indices[(m - 1) % 8]
        T[m, m] += p_continue
        T[m, left_idx] += p_turn_each
        T[m, right_idx] += p_turn_each
        T[m, stop_idx] += p_stop

    # Transition probabilities from stationary mode (stay stopped or resume moving)
    T[stop_idx, stop_idx] = 0.60
    for m in move_indices:
        T[stop_idx, m] = 0.40 / len(move_indices)

    # Normalize rows to valid probability distributions
    row_sums = T.sum(axis=1, keepdims=True)
    return T / np.maximum(row_sums, 1e-12)


class MarkovOccupancyPredictor:
    """Grid-based Markov occupancy predictor over an 8m window around the
    robot"""

    def __init__(self, grid_res=0.25, window_size_m=8.0, dt=0.1, horizon_s=3.0, speed_mps=0.8, pos_sigma_m=0.35, transition_matrix=None,):
        # Grid and time-horizon setup
        self.grid_res = float(grid_res)
        self.window_size_m = float(window_size_m)
        self.half_window = 0.5 * self.window_size_m
        self.n_cells = int(round(self.window_size_m / self.grid_res))
        self.dt = float(dt)
        self.horizon_steps = max(1, int(round(horizon_s / self.dt)))
        self.speed_mps = float(speed_mps)
        self.pos_sigma_m = float(pos_sigma_m)
        self.T = (build_default_transition_matrix() if transition_matrix is None else np.asarray(transition_matrix, dtype=float))

        # Active pedestrian tracks keyed by track ID
        self.tracks = {}
        self._last_robot_xy = (0.0, 0.0)

    def _world_to_local_idx(self, x, y, robot_x, robot_y):
        """Convert world coordinates into indices of 8x8m window centered
        on the robot"""
        gx = int(math.floor((x - (robot_x - self.half_window)) / self.grid_res))
        gy = int(math.floor((y - (robot_y - self.half_window)) / self.grid_res))
        return gx, gy

    def _init_belief(self, ped_x, ped_y, vx, vy, robot_x, robot_y):
        """Seed belief near current pedestrian position with mild bias toward
        observed velocity"""
        belief = np.zeros((self.n_cells, self.n_cells, len(MOTION_MODES)), dtype=float)
        gx, gy = self._world_to_local_idx(ped_x, ped_y, robot_x, robot_y)
        if 0 <= gx < self.n_cells and 0 <= gy < self.n_cells:
            speed = math.hypot(vx, vy)
            if speed > 1e-3:
                vel_dir = np.array([vx / speed, vy / speed])
                mode_scores = np.maximum(0.1, _MOTION_DIRS @ vel_dir)
            else:
                mode_scores = np.array([0.05] * 8 + [0.69])
            mode_probs = mode_scores / np.sum(mode_scores)
            belief[gy, gx, :] = mode_probs
        return belief

    def _bayes_update(self, belief, obs_x, obs_y, vx, vy, robot_x, robot_y):
        """Combines Gaussian position likelihood with velocity-mode
        likelihood """
        # Computes position likelihood over all grid cells
        yy, xx = np.indices((self.n_cells, self.n_cells), dtype=float)
        cell_x = (robot_x - self.half_window) + (xx + 0.5) * self.grid_res
        cell_y = (robot_y - self.half_window) + (yy + 0.5) * self.grid_res
        dist2 = (cell_x - obs_x) ** 2 + (cell_y - obs_y) ** 2
        pos_like = np.exp(-0.5 * dist2 / (self.pos_sigma_m ** 2))

        # Computes directional mode likelihood from observed velocity
        speed = math.hypot(vx, vy)
        if speed > 0.05:
            vel_dir = np.array([vx / speed, vy / speed])
            mode_like = np.exp(2.0 * (_MOTION_DIRS @ vel_dir))
        else:
            mode_like = np.array([0.2] * 8 + [2.0])

        # Multiply prior by likelihoods and normalize posterior
        posterior = belief * pos_like[:, :, None] * mode_like[None, None, :]
        total = np.sum(posterior)
        if total <= 1e-12:
            return self._init_belief(obs_x, obs_y, vx, vy, robot_x, robot_y)
        return posterior / total

    def _propagate_one_step(self, belief, rem_xy=None):
        """Shift probability mass along each motion mode, then apply
        Markov transtion T^T"""
        shifted = np.zeros_like(belief)
        step_cells = (self.speed_mps * self.dt) / self.grid_res

        if rem_xy is None:
            rem_xy = np.zeros((len(MOTION_MODES), 2), dtype=float)

        # Advects each mode's probability plane by accumulated integer step offset
        for m, (dx_m, dy_m) in enumerate(MOTION_MODES):
            rem_xy[m, 0] += dx_m * step_cells
            rem_xy[m, 1] += dy_m * step_cells
            shift_x = int(round(rem_xy[m, 0]))
            shift_y = int(round(rem_xy[m, 1]))
            rem_xy[m, 0] -= shift_x
            rem_xy[m, 1] -= shift_y

            plane = belief[:, :, m]
            if not np.any(plane):
                continue
            src_x0 = max(0, -shift_x)
            src_x1 = min(self.n_cells, self.n_cells - shift_x)
            dst_x0 = max(0, shift_x)
            dst_x1 = min(self.n_cells, self.n_cells + shift_x)

            src_y0 = max(0, -shift_y)
            src_y1 = min(self.n_cells, self.n_cells - shift_y)
            dst_y0 = max(0, shift_y)
            dst_y1 = min(self.n_cells, self.n_cells + shift_y)

            if src_x1 > src_x0 and src_y1 > src_y0:
                shifted[dst_y0:dst_y1, dst_x0:dst_x1, m] = plane[src_y0:src_y1, src_x0:src_x1]

        # Mixes modes using the Markov transition matrix
        # and re-normalizes
        propagated = np.tensordot(shifted, self.T.T, axes=([-1], [0]))
        total = np.sum(propagated)
        return propagated / total if total > 1e-12 else belief

    def update_and_predict(self, observations, robot_pose, dt=None):
        """Updates pedestrians tracks and returns a list of occupancy heatmaps a_l(c)
        for l = 1. horizon_steps"""
        step_dt = self.dt if dt is None else float(dt)
        robot_x, robot_y = float(robot_pose[0]), float(robot_pose[1])
        self._last_robot_xy = (robot_x, robot_y)
        obs_list = list(observations or [])

        # Update matched tracks or initialize new ones
        matched_ids = set()
        for idx, obs in enumerate(obs_list):
            ox, oy = float(obs["x"]), float(obs["y"])
            ped_id = obs.get("id", f"ped_{idx}")

            if ped_id in self.tracks:
                prev = self.tracks[ped_id]
                vx = float(obs.get("vx", (ox - prev["pos"][0]) / max(step_dt, 1e-3)))
                vy = float(obs.get("vy", (oy - prev["pos"][1]) / max(step_dt, 1e-3)))

                prior = self._propagate_one_step(prev["belief"], prev["rem_xy"])
                belief = self._bayes_update(prior, ox, oy, vx, vy, robot_x, robot_y)
                self.tracks[ped_id] = {"pos": np.array([ox, oy]), "vel": np.array([vx, vy]), "belief": belief, "rem_xy": prev["rem_xy"],}
            else:
                vx = float(obs.get("vx", 0.0))
                vy = float(obs.get("vy", 0.0))
                belief = self._init_belief(ox, oy, vx, vy, robot_x, robot_y)
                self.tracks[ped_id] = {"pos": np.array([ox, oy]), "vel": np.array([vx, vy]), "belief": belief, "rem_xy": np.zeros((len(MOTION_MODES), 2), dtype=float),}

            matched_ids.add(ped_id)

        # Open-Loop propagation for tracks without a current observation
        for ped_id, track in list(self.tracks.items()):
            if ped_id not in matched_ids:
                track["belief"] = self._propagate_one_step(track["belief"], track["rem_xy"])

        # Roll out multi-step occupancy heatmaps across all
        # tracked pedestrians
        heatmaps = []
        current_beliefs = {pid: (tr["belief"].copy(), tr["rem_xy"].copy()) for pid, tr in self.tracks.items()}
        for _ in range(self.horizon_steps):
            combined_occ = np.zeros((self.n_cells, self.n_cells), dtype=float)
            for pid, (b_step, rem_step) in current_beliefs.items():
                next_b = self._propagate_one_step(b_step, rem_step)
                current_beliefs[pid] = (next_b, rem_step)

                occ = np.clip(np.sum(next_b, axis=-1), 0.0, 1.0)

                combined_occ = 1.0 - (1.0 - combined_occ) * (1.0 - occ)
            heatmaps.append(combined_occ)

        return heatmaps

    def query_max_occupancy(self, xs, ys, horizon_idx, heatmaps):
        """Return predicted occupancy probabilities at (xs, ys) for a
        given horizon step."""
        if not heatmaps:
            return np.zeros_like(xs, dtype=float)
        safe_l = min(max(0, int(horizon_idx)), len(heatmaps) - 1)
        occ_grid = heatmaps[safe_l]
        robot_x, robot_y = self._last_robot_xy

        # Max query coordinates into local occupancy grid indices
        gx = np.floor((xs - (robot_x - self.half_window)) / self.grid_res).astype(np.int32)
        gy = np.floor((ys - (robot_y - self.half_window)) / self.grid_res).astype(np.int32)
        in_bounds = (gx >= 0) & (gx < self.n_cells) & (gy >= 0) & (gy < self.n_cells)

        sampled = np.zeros_like(xs, dtype=float)
        valid = np.where(in_bounds)
        sampled[valid] = occ_grid[gy[valid], gx[valid]]
        return sampled


class MDWBLocalPlanner(DWBLocalPlanner):
    """
    DWB local planner with Markov pedestrian-risk cost
    (w_risk = 0.0 acts like standard DWB)
    """

    def __init__(self, grid, resolution, origin_x, origin_y, predictor=None, discount_gamma=0.95, w_risk=25.0, **kwargs,):
        kwargs["w_risk"] = float(w_risk)
        super().__init__(grid, resolution, origin_x, origin_y, **kwargs)
        self.predictor = MarkovOccupancyPredictor() if predictor is None else predictor
        self.discount_gamma = float(discount_gamma)
        self._latest_heatmaps = []
        self._last_robot_xy = (0.0, 0.0)

    def _compute_markov_risk(self, xs, ys, heatmaps):
        """
        Computes discounted pedestrian occupancy risk
        along candidate trajectories (xs, ys)
        """
        num_cand, n_steps = xs.shape
        risk = np.zeros(num_cand, dtype=float)
        if self.w_risk <= 0.0 or not heatmaps:
            return risk

        robot_x, robot_y = self._last_robot_xy
        half_w = self.predictor.half_window
        g_res = self.predictor.grid_res
        n_cells = self.predictor.n_cells
        pred_dt = max(self.predictor.dt, 1e-6)

        # Sample center plus cardinal footprint points around the robot
        r_fp = self.robot_radius
        fp_offsets = [(0.0, 0.0), (r_fp, 0.0), (-r_fp, 0.0), (0.0, r_fp), (0.0, -r_fp)]

        # Accumulate discounted worst-case occupancy along each candidate rollout
        for i in range(n_steps):
            t_i = (i + 1) * self.sim_granularity
            l_i = min(max(0, int(round(t_i / pred_dt)) - 1), len(heatmaps) - 1)
            occ_grid = heatmaps[l_i]

            cell_max_occ = np.zeros(num_cand, dtype=float)
            for ox, oy in fp_offsets:
                px = xs[:, i] + ox
                py = ys[:, i] + oy
                gx = np.floor((px - (robot_x - half_w)) / g_res).astype(np.int32)
                gy = np.floor((py - (robot_y - half_w)) / g_res).astype(np.int32)
                valid = (gx >= 0) & (gx < n_cells) & (gy >= 0) & (gy < n_cells)
                if np.any(valid):
                    vals = np.zeros(num_cand, dtype=float)
                    vals[valid] = occ_grid[gy[valid], gx[valid]]
                    cell_max_occ = np.maximum(cell_max_occ, vals)

            risk += (self.discount_gamma ** (i + 1)) * cell_max_occ

        return risk

    def compute_velocity_commands(self ,current_pose, current_vel, global_path, dt=0.1, last_waypoint_idx=0, pedestrian_observations=None,):
        """Pick (v, omega) using DWB costs plus Markov risk, updating
        the forecast when observations arrive."""
        x, y, th = current_pose
        curr_v, curr_w = current_vel
        self._last_robot_xy = (float(x), float(y))

        # Refresh pedestrian occupancy forecast when new observations arrive
        if pedestrian_observations is not None:
            self._latest_heatmaps = self.predictor.update_and_predict(pedestrian_observations, current_pose, dt=dt)

        # Fall back to standard DWB when risk weighting or heatmaps are inactive
        if self.w_risk <= 0.0 or not self._latest_heatmaps:
            return super().compute_velocity_commands(current_pose, current_vel, global_path, dt=dt, last_waypoint_idx=last_waypoint_idx)

        # Goal check and local target selection along the global path
        goal_x, goal_y = global_path[-1]
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)
        if dist_to_goal <= self.goal_tolerance:
            return 0.0, 0.0, {"reached": True, "dist_to_goal": dist_to_goal, "waypoint_idx": len(global_path) - 1,}

        v_min, v_max, w_min, w_max = self._compute_dynamic_window(curr_v, curr_w, dt)

        search_start = max(0, last_waypoint_idx - 2)
        search_end = min(len(global_path), search_start + 40)
        path_arr = np.array(global_path[search_start:search_end])
        dists = np.hypot(path_arr[:, 0] - x, path_arr[:, 1] - y)
        closest_idx = search_start + int(np.argmin(dists))

        lookahead_idx = min(len(global_path) - 1, closest_idx + 4)
        target_pt = global_path[lookahead_idx]
        target_heading = math.atan2(target_pt[1] - y, target_pt[0] - x)
        heading_err = abs(math.atan2(math.sin(target_heading - th), math.cos(target_heading - th)))

        # Sample candidate linear/angular velocities and roll out trajectories
        if self.cruise_speed_mps is not None:
            if heading_err <= 0.35:
                v_cand = np.linspace(max(self.min_vel_x, 0.0), self.cruise_speed_mps, max(3, min(self.vx_samples, 5)))
            else:
                v_cand = np.array([0.0])
        else:
            effective_v_max = v_max * max(0.1, math.cos(min(heading_err, math.pi / 2)))
            v_cand = np.linspace(v_min, effective_v_max, self.vx_samples)

        w_cand = np.linspace(w_min, w_max, self.vtheta_samples)
        V_grid, W_grid = np.meshgrid(v_cand, w_cand)
        V = V_grid.flatten()
        W = W_grid.flatten()

        t_arr = np.arange(1, self.sim_steps + 1) * self.sim_granularity
        ths = th + W[:, None] * t_arr[None, :]
        xs = x + np.cumsum(V[:, None] * np.cos(ths) * self.sim_granularity, axis=1)
        ys = y + np.cumsum(V[:, None] * np.sin(ths) * self.sim_granularity, axis=1)

        # Filter out trajectories that leave map bounds or hit static obstacles
        gxs = ((xs - self.origin_x) / self.res).astype(np.int32)
        gys = ((ys - self.origin_y) / self.res).astype(np.int32)
        out_of_bounds = (gxs < 0) | (gxs >= self.map_w) | (gys < 0) | (gys >= self.map_h)
        safe_gx = np.clip(gxs, 0, self.map_w - 1)
        safe_gy = np.clip(gys, 0, self.map_h - 1)
        collided = np.any(out_of_bounds | self.blocked[safe_gy, safe_gx], axis=1)

        valid_idx = np.where(~collided)[0]
        if len(valid_idx) == 0:
            err = math.atan2(target_pt[1] - y, target_pt[0] - x) - th
            err = math.atan2(math.sin(err), math.cos(err))
            cmd_w = float(np.clip(err * 1.5, self.min_vel_theta, self.max_vel_theta))
            return 0.0, cmd_w, {"reached": False, "dist_to_goal": dist_to_goal, "waypoint_idx": closest_idx,}

        # Score valid rollouts with DWB critics plus Markov risk and pick minimum
        end_x = xs[valid_idx, -1]
        end_y = ys[valid_idx, -1]
        end_th = ths[valid_idx, -1]
        cand_v = V[valid_idx]

        path_dist = np.hypot(target_pt[0] - end_x, target_pt[1] - end_y)
        goal_dist = np.hypot(goal_x - end_x, goal_y - end_y)
        angle_diff = np.abs(np.arctan2(np.sin(end_th - target_heading), np.cos(end_th - target_heading)))
        speed_cost = (self.max_vel_x - cand_v) / max(self.max_vel_x, 1e-3)
        risk_cost = self._compute_markov_risk(xs[valid_idx], ys[valid_idx], self._latest_heatmaps)

        costs = (self.w_path * path_dist + self.w_goal * goal_dist + self.w_align * angle_diff + self.w_speed * speed_cost + self.w_risk * risk_cost)

        best_local = int(np.argmin(costs))
        best_i = valid_idx[best_local]
        return float(V[best_i]), float(W[best_i]), {"reached": False, "dist_to_goal": dist_to_goal, "waypoint_idx": closest_idx, "markov_risk": float(risk_cost[best_local]),}