# Prediction-Aware Autonomous Navigation

Level 1 compares the project A* and RRT global planners with the local DWB controller in a live ROS 2 and Gazebo simulation. Watch the robot in the Gazebo GUI; the launch does not open RViz.

## Level 1 behavior

- The 12 x 12 m arena, ground, and four perimeter walls stay fixed for the whole run.
- Each trial moves and rotates the fixed 2 x 2 m center obstacle. The occupancy map used by A* and RRT is regenerated with the same pose.
- Each trial moves the green start pad and red goal pad. Before each planner run, the robot is teleported to the center of the start pad. DWB's differential-drive poses are rendered directly in Gazebo so the visible robot stays synchronized with the selected global path.
- A* and RRT start from the same pose. A* searches the occupancy grid; RRT paths are shortened where collision-free segments are available.
- Both planner runs use the same 0.35 m/s moving cruise speed. DWB may use 0 m/s briefly while turning in place to align with the next path segment.
- Results are written to `level1_results.csv` and `level1_results.json`.
- After both planners finish a trial, the terminal pauses. Press `x` to set up the next trial, so the completed scene stays visible in Gazebo.

## Run the GUI evaluation

Terminal 1 - from the ROS workspace root, build and launch Gazebo:

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select vehicle_description
source install/setup.bash
ros2 pkg prefix vehicle_description
ros2 launch vehicle_description level1_sim.launch.py
```

Terminal 2 - run 30 paired trials:

```bash
cd ~/ros2_ws
source install/setup.bash
python3 src/PAAN/run_level1.py --ros2 --trials 30
```

Use `--trials 1` for a quick visual check. `--seed N` selects a repeatable sequence, and `--max_time T` sets the time limit per planner run.
