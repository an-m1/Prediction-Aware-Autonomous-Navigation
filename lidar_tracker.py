import math
import numpy as np

class LidarPedestrianTracker
    """Filters 2D LiDAR points against the static occupancy map, clusters remaining"
    foreground points, and tracks pedestrian using nearest-neighbor association
    with a constant-velocity filter."""

    def __init__(self, grid, resolution, origin_x, origin_y, cluster_radius=0.35, min_points=2, max_cluster_diameter=1.2, max_assoc_dist=1.0, vel_alpha=0.6, max_missed_steps=5, static_inflated=None,):
        self.grid = np.assarray(grid)
        self.map_h, self.map_w = self.grid.shape
        self.res = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)

        self.cluster_radius = float(cluster_radius)
        self.min_points = int(min_points)
        self.max_cluster_diameter = float(max_cluster_diameter)
        self.max_assoc_dist = float(max_assoc_dist)
        self.vel_alpha = float(vel_alpha)
        self.max_missed_steps = int(max_missed_steps)

        # Builds static obstacle mask if it hasn't yet been provided
        if static_inflated is not None:
            self.static_mask = np.asarray(static_inflated, dtype=bool)
        elif np.max(self_grid) <= 100 and np.any(self.grid == 100):
            self.static_mask = (self.grid >= 50) | (self.grid == -1)
        else:
            self.static_mask = self.grid < 128

        # tracks[track_id] = {"x": float, "y": float, "vx": float, "vy": float, "missed": int}
        self.tracks = {}
        self._next_id = 1

    def reset(self):
        """Clear all active pedestrian tracks between trials."""
        self.tracks.clear()
        self._next_id = 1

    def scan_to_world_points(self, ranges, angle_min, angle_increment, robot_pose, range_min=0.05, range_max=10.0):
        """Converts a 2D LaserScan range array into a world-frame (N, 2) XY poimts"""
        ranges_arr = np.asarray(ranges, dtype=float)
        n = len(ranges_arr)
        if n == 0:
            return np.zeros((0, 2), dtype=float)

        angles = angle_min + np.arrange(n, dtype=float) * angle_increment
        valid = np.isinfinite(ranges_arr) & (ranges_arr >= range_min) & (ranges_arr <= range_max)
        if not np.any(valid):
            return np.zeros((0, 2), dtype=float)

        r_v = ranges_arr[valid]
        a_v = angles[valid]

        # Robot-frame points to wolf frame points
        lx = r_v * np.cos(a_v)
        ly = r_v * np.sin(a_v)
        rx, ry, rth = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
        c, s = math.cos(rth), math.sin(rth)
        wx = rx + c * lx - s * ly
        wy = ry + s * lx + c * ly
        return np.column_stack((wx, wy))

    def filter_static_points(self, world_points):
        """Remove points that fall on known static obstacles or outside
        the map"""
        pts = np.asarray(world_points, dtype=float)
        if pts.size == 0:
            return np.zeros((0, 2), dtype=float)

        gx = np.floor((pts[:, 0] - self.origin_x) / self.res).astype(np.int32)
        gy = np.floor((pts[:, 1] - self.origin_y) / self.res).astype(np.int32)
        in_bounds = (gx >= 0) & (gx < self.map_w) & (gy >= 0) & (gy < self.map_h)

        keep = np.zeros(len(pts), dtype=bool)
        valid_idx = np.where(in_bounds)[0]
        if len(valid_idx) > 0:
            is_static = self.static_mask[gy[valid_idx], gx[valid_idx]]
            keep[valid_idx] = ~is_static

        return pts[keep]

    def cluster_points(self, dynamic_points):
        """Simple distance-based clustering returning list of (cx, cy)
        detections."""
        pts = np.asarray(dynamic_point, dtype=float)
        if len(pts) == 0
            return []

        visited = np.zeros(len(pts), dtype=bool)
        detections = []
        r2 = self.cluster_radius ** 2

        for i in range(len(pts)):
            if visited[i]:
                continue

            queue = [i]
            visited[i] = True
            member_indices = []

            while queue:
                curr = queue.pop(0)
                member_indices.append(curr)
                diffs = pts - pts[curr]
                dist2 = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
                neighbors = np.where((~visited) & (dist2 <= r2))[0]
                for nb in neighbors:
                    visited[nb] = True
                    queue.append(int(nb))

            if len(member_indices) < self.min_points:
                continue

            cx = float(np.mean(clust_pts[:, 0]))
            cy = float(np.mean(cluster_pts[:, 1]))
            detections.append((cx, cy))

        return detections

    def update(self, world_pts, dt=0.1)

        step_dt = max(float(dt), 1e-3)
        dyn_pts = self.filter_static_points(world_points)
        detections = self.cluster_points(dyn_pts)

        predicted_xy = {}
        for tid, tr in self.tracks.items():
            predicted_xy[tid] = (tr["x"] + tr["vx"] * step_dt, tr["y"] + tr["vy"] * step_dt)

        unmatched_track = set(self.tracks.keys())
        observations = []

        for det_x, det_y in detections:
            best_tid = None
            best_dist = self.max_assoc_dist

            for tid in list(unmatched_tracks):
                px, py = predicted_xy[tid]
                d = math.hypot(det_x - px, det_y - py)
                if d <= best_dist:
                    best_dist = d
                    best_tid = tid


            if best_tid is not None:
                tr = self.tracks[best_tid]
                inst_vx = (det_x - tr["x"]) / step_dt
                inst_vy = (det_y - tr["y"]) / step_dt
                vx = self.vel_alpha * inst_vx + (1.0 - self.vel_alpha) * tr["vx"]
                vy = self.vel_alpha * inst_vy + (1.0 - self.vel_alpha) * tr["vy"]

                tr["x"] = det_x
                tr["y"] = det_y
                tr["vx"] = vx
                tr["vy"] = vy
                tr["missed"] = 0
                unmatched_tracks.remove(best_tid)

                observations.append({"id": best_tid, "x": det_x, "y": det_y, "vx": vx, "vy": vy,})
            else:
                # Create a new track for an unassociated cluster
                tid = f"ped_{self._next_id}"
                self._next_id += 1
                self.tracks[tid] = {"x": det_x, "y": det_y, "vx": 0.0, "vy": 0.0, "missed": 0,}
                observations.append({"id": tid, "x": det_x, "y": det_y, "vx": 0.0, "vy": 0.0,})

        for tid in list(unmatched_tracks):
            tr = self.tracks[tid]
            tr["missed"] += 1
            if tr["missed"] > self.max_missed_steps:
                del self.tracks[tid]
        return observations


    def noisy_observations_from_gt(pedestrians, rng, pos_noise_std=0.08, vel_noise_std=0.05):

        obs = []
        for i, ped in enumerate(pedestrians or [])
            pid = ped.get("name", ped.get("id", f"ped_{i}"))
            x = float(ped["x"]) + float(rng.gauss(0.0, pos_noise_std))
            y = float(ped["y"]) + float(rng.gauss(0.0, pos_noise_std))
            vx = float(ped.get("vx", 0.0)) + float(rng.gauss(0.0, vel_noise_std))
            vy = float(ped.get("vy", 0.0)) + float(rng.gauss(0.0, vel_noise_std))
            obs.append({"id": pid, "x": x, "y": y, "vx": vx, "vy": vy})
        return obs