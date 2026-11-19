#!/usr/bin/env python3
"""Live top-down map for Level 3 dynamic-obstacle trials."""

import rclpy

from level2_visualizer import Level1Visualizer


class Level3Visualizer(Level1Visualizer):
    LEVEL = 3
    TOPIC_PREFIX = "/level3/visualization"


def main():
    rclpy.init()
    node = Level3Visualizer()
    try:
        node.root.mainloop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
