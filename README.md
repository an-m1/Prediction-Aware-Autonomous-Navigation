# Prediction-Aware Autonomous Navigation

Level 1 compares the project A* and RRT global planners with the local DWB controller in a live ROS 2 and Gazebo simulation. Watch the robot in the Gazebo GUI; the launch does not open RViz.

## Level 1 behavior

- The 12 x 12 m arena, ground, and four perimeter walls stay fixed for the whole run.
- Each trial randomly selects one of five center obstacle sizes from 1.5 m to 2.5 m, then moves and rotates it. The occupancy map and Gazebo collision box use the same dimensions.
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

Terminal 3 - open the live top-down planner map before starting Terminal 2:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch vehicle_description level1_visualizer.launch.py
```

The map uses a white background and shows obstacles, start and goal markers,
explored A* graph edges and RRT tree branches, final A* and RRT paths, and the
live robot heading with its trajectory trail.

When all trials finish, `level1_trial_grid.svg` is saved beside
`level1_results.csv` and `level1_results.json`. It contains a top-down grid of
every trial with the obstacle, start/goal markers, planned paths, recorded A*
and RRT trajectories, and success states.

Use `--trials 1` for a quick visual check. `--seed N` selects a repeatable sequence, and `--max_time T` sets the time limit per planner run.

## Level 2: randomized multiple static obstacles

Level 2 reuses `astar.py`, `rrt.py`, and `dwb.py` with maps containing scattered
rectangular obstacles. The committed `level2_map.pgm` and `level2_world.sdf`
show a six obstacle example. The automated evaluator creates a new layout for
each paired trial at moderate and dense clutter. Moderate layouts randomly use
3 or 4 obstacles; dense layouts randomly use 5 or 6. Each obstacle is sampled
from a pool of ten varied box sizes (0.75 m to 1.8 m), matching the Gazebo
collision boxes. Scenario generation and RRT use explicit seeds so runs can be
reproduced.

Run the automated evaluation from the ROS workspace source directory after
building the package and sourcing the workspace:

```bash
python3 src/PAAN/run_level2.py --trials 30 --seed 12345
```

To see the randomized Level 2 trial maps and planner paths in a Python window,
start the visualizer in one sourced terminal:

```bash
ros2 launch vehicle_description level2_visualizer.launch.py
```

In another sourced terminal run the evaluator with `--ros2`; it publishes each
scenario and robot trajectory to that Python map window:

```bash
python3 src/PAAN/run_level2.py --trials 3 --ros2
```

The Python map visualizer is separate from Gazebo. Start `level2_sim.launch.py`
in a third terminal to open the example world; keep it running while the
evaluator runs so the robot and obstacles can be moved through Gazebo. The
evaluator runs the same DWB kinematic model as Level 1 and renders each pose in
Gazebo. It writes paired trial data,
planner sensitivity sweeps (A* resolutions 0.05/0.10/0.20 m; RRT step size,
goal bias, and iteration budget), and 10 RRT seeds on each of the first 10
layouts to `level2_results.csv`, `level2_results.json`,
`level2_sensitivity.csv`, and `level2_rrt_repeated_seeds.csv`. The JSON summary
includes per-clutter success rates, median/IQR metric distributions, exact
McNemar p-values and paired Wilcoxon tests when SciPy is installed. CSVs retain
individual outcomes for failure analysis (`planning_failed`, `collision`, or
`timeout`). Use `--output-dir PATH` to choose another output directory.

The separate Gazebo example opens the committed layout; the automated
evaluator generates independent layouts for its statistical runs.
