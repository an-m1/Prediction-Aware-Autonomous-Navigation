import math
import time
import subprocess

WORLD = "level3_world"
UPDATE_RATE = 20

def move_pedestrians(positions):

    poses = []

    for name, x, y in positions:

        poses.append(
            f'pose {{ name: "{name}" '
            f'position {{ x: {x:.4f} y: {y:.4f} z: 0.85 }} '
            f'orientation {{ w: 1 }} }}'
        )

    subprocess.run(
        [
            "gz", "service",
            "-s", f"/world/{WORLD}/set_pose_vector",
            "--reqtype", "gz.msgs.Pose_V",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000",
            "--req", " ".join(poses)
        ],
        stdout=subprocess.DEVNULL
    )


def main():

    print("Level 3 pedestrian movement started")
    print("Press Ctrl+C to stop")

    start_time = time.monotonic()

    try:

        while True:

            loop_start = time.monotonic()
            t = loop_start - start_time

            x1 = -3.0
            y1 = 2.0 * math.sin(0.3 * t)

            x2 = 3.0 + 1.8 * math.sin(0.35 * t)
            y2 = 3.0

            x3 = 1.5
            y3 = -1.5 + 2.5 * math.sin(0.25 * t)

            positions = [
                ("pedestrian_1", x1, y1),
                ("pedestrian_2", x2, y2),
                ("pedestrian_3", x3, y3)
            ]

            move_pedestrians(positions)

            elapsed = time.monotonic() - loop_start

            time.sleep(
                max(0.0, 1.0 / UPDATE_RATE - elapsed)
            )

    except KeyboardInterrupt:

        print("\nPedestrian movement stopped")


if __name__ == "__main__":
    main()
