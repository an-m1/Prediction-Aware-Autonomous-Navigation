import argparse
import math
import time
import subprocess

# Gazebo world and loop frequency settings
WORLD = "level3_world"
UPDATE_RATE = 20

def move_pedestrians(positions, world=WORLD):
    """Update Gazebo pedestrian model poses using Pose_V service"""
    
    # Build protobuf Pose_V message entries for each pedestrian
    poses = []
    for name, x, y in positions:
        poses.append(
            f'pose {{ name: "{name}" '
            f'position {{ x: {x:.4f} y: {y:.4f} z: 0.85 }} '
            f'orientation {{ w: 1 }} }}'
        )

    # Send pose updates to Gazebo via CLI service call
    subprocess.run(
        [
            "gz",
            "service",
            "-s",
            f"/world/{world}/set_pose_vector",
            "--reqtype",
            "gz.msgs.Pose_V",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "1000",
            "--req",
            " ".join(poses),
        ],
        stdout=subprocess.DEVNULL
    )


def _follow_waypoints(waypoints, elapsed, speed=0.65, stop_at_idx=None, stop_duration=0.0):
    """Compute (x, y, vx, vy) along the polyline with optional waypoint
        pause."""
    pts = [tuple(p) for p in waypoints]
    if not pts:
        return 0.0, 0.0, 0.0, 0.0
    if len(pts) == 1:
        return pts[0][0], pts[0][1], 0.0, 0.0

    rem_t = max(0.0, float(elapsed))
    spd = max(float(speed), 1e-4)

    # Advance through segments and handle optional stop at a target waypoint
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        seg_time = seg_len / spd

        if rem_t <= seg_time:
            frac = rem_t / max(seg_time, 1e-9)
            ux, uy = (x1 - x0) / max(seg_len, 1e-9), (y1 - y0) / max(seg_len, 1e-9)
            return x0 + frac * (x1 - x0), y0 + frac * (y1 - y0), spd * ux, spd * uy

        rem_t -= seg_time

        # Optional pause when reaching a specific waypoint index
        if stop_at_idx is not None and (i + 1) == stop_at_idx and stop_duration > 0.0:
            if rem_t <= stop_duration:
                return x1, y1, 0.0, 0.0
            rem_t -= stop_duration

    # Hold final waypoint once reached
    return pts[-1][0], pts[-1][1], 0.0, 0.0

def evaluate_pedestrians(elapsed, mode="sine", scripts=None):
    """Return list of dicts: {'id': name, 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'radius': 0.25}.
    Supports scripted scenarios ('a', 'b', 'c' or custom scripts) plus original 'sine' mode."""
    
    # Custom script definitions
    if scripts:
        out = []
        for idx, sc in enumerate(scripts):
            pid = sc.get("id", f"pedestrian_{idx + 1}")
            x, y, vx, vy = _follow_waypoints(sc.get("waypoints", []), elapsed, speed=sc.get("speed", 0.65), stop_at_idx=sc.get("stop_at_idx"), stop_duration=sc.get("stop_duration", 0.0),)
            
            out.append({"id": pid, "x": x, "y": y, "vx": vx, "vy": vy, "radius": float(sc.get("radius", 0.25)),})
        return out

    # Built-in scripted scenarios ('a', 'b', 'c')
    if mode == "a":
        x, y, vx, vy = _follow_waypoints([(-3.5, 0.0), (3.5, 0.0)], elapsed, speed=0.65)
        return [{"id": "pedestrian_1", "x": x, "y": y, "vx": vx, "vy": vy, "radius": 0.25}]

    if mode == "b":
        # Pedestrian crosses and pauses mid-corridor on the planned path
        x, y, vx, vy = _follow_waypoints(
            [(-3.5, 0.0), (0.0, 0.0), (3.5, 0.0)], elapsed, speed=0.65, stop_at_idx=1, stop_duration=3.0,)
        return [{"id": "pedestrian_1", "x": x, "y": y, "vx": vx, "vy": vy, "radius": 0.25}]

    if mode == "c":
        # Two pedestrians with different speeds/timing and a brief pause
        x1, y1, vx1, vy1 = _follow_waypoints(
            [(-3.5, -1.0), (0.0, 0.2), (3.5, 1.2)], elapsed, speed=0.70, stop_at_idx=1, stop_duration=1.5,)
        x2, y2, vx2, vy2 = _follow_waypoints([(3.0, -2.5), (-2.5, 2.5)], max(0.0, elapsed - 0.8), speed=0.55,)
        return [{"id": "pedestrian_1", "x": x1, "y": y1, "vx": vx1, "vy": vy1, "radius": 0.25}, {"id": "pedestrian_2", "x": x2, "y": y2, "vx": vx2, "vy": vy2, "radius": 0.25},]

    # Default sinusoidal trajectories for three pedestrians
    x1, y1 = -3.0, 2.0 * math.sin(0.3 * elapsed)
    vx1, vy1 = 0.0, 2.0 * 0.3 * math.cos(0.3 * elapsed)

    x2, y2 = 3.0 + 1.8 * math.sin(0.35 * elapsed), 3.0
    vx2, vy2 = 1.8 * 0.35 * math.cos(0.35 * elapsed), 0.0

    x3, y3 = 1.5, -1.5 + 2.5 * math.sin(0.25 * elapsed)
    vx3, vy3 = 0.0, 2.5 * 0.25 * math.cos(0.25 * elapsed)

    return [{"id": "pedestrian_1", "x": x1, "y": y1, "vx": vx1, "vy": vy1, "radius": 0.25}, {"id": "pedestrian_2", "x": x2, "y": y2, "vx": vx2, "vy": vy2, "radius": 0.25}, {"id": "pedestrian_3", "x": x3, "y": y3, "vx": vx3, "vy": vy3, "radius": 0.25},]

def main():
    """Run CLI loop to update Gazebo pedestrian poses in the selected scenario mode."""
    parser = argparse.ArgumentParser(description="Move Gazebo pedestrians for Level 3.")
    parser.add_argument("--world", default=WORLD)
    parser.add_argument("--rate", type=float, default=UPDATE_RATE)
    parser.add_argument("--mode", choices=["sine", "a", "b", "c"], default="sine")
    args = parser.parse_args()

    print(f"Level 3 pedestrian movement started (world={args.world}, mode={args.mode})")
    print("Press Ctrl+C to stop")

    start_time = time.monotonic()
    period = 1.0 / max(args.rate, 1e-3)

    try:
        # Evaluate pedestrian states and push pose updates at a fixed period
        while True:
            loop_start = time.monotonic()
            t = loop_start - start_time

            states = evaluate_pedestrians(t, mode=args.mode)
            positions = [(s["id"], s["x"], s["y"]) for s in states]
            move_pedestrians(positions, world=args.world)

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\nPedestrian movement stopped")


if __name__ == "__main__":
    main()