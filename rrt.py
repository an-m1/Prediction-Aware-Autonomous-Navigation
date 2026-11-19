import math
import random
import numpy as np

# RRT grows a tree of connected points to navigate around obstacles. Every point in that tree is represented
# by a node object.

class Node:
    def _init_(self, x, y):
        # stores (x, y) in meters
        self.x = x
        self.y = y
        # A pointer back to the original node that connected to this one.
        # vital since once goal is reached parents are used to trace path back from goal to start
        self.parent = None

class RRTPlanner:
    """Sets up the planner by loading the occupancy grid map, dimensions, metric resolution (0.05 m/px),
    map origin, and the robot's physical radius (0.18 m being used as a safety buffer)"""
    def _init_(self, grid, resolution, origin_x, origin_y, robot_radius=0.18):
        self,grid = np.array(grid)
        self.map_h, self.map_w = self.grid.shape
        self.res = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.inflation_px = math.ceil(robot_radius / self.res) # Converts robot's physical radius to pixels to know how much buffer space to keep away from walls and obstacles
        self.blocked = self._compute_inflated_grid() # Pre-calculates an obstacle mask ahead of time so the planner doesn't have to recalculate safety boundaries during every iteration

    def _world_to_grid(self, x_m, y_m):
        gx = int((x_m - self.origin_x) / self.res)
        gy = int((y_m - self.origin_y) / self.res)
        return gx, gy

    def _world_to_grid(self, x_m, y_m):
        gx = int((x_m - self.origin_x) / self.res)
        gy = int((y_m - self.origin_y) / self.res)
        return gx, gy

    def _compute_inflated_grid(self):
        """
        Finds every obstacle pixel (grid == 0) and puffs it outward by a circle of radius self.inflation_px
        treats the robot as a single point moving on an inflated map, ensuring the robot's physical edges won't scrape against the walls
        """
        blocked = np.zeroes((self.map_h, self.map_w), dtype=bool)
        obs_y, obs_x = np.where(self.grid == 0)
        For oy, ox in zip(obs_y, obs_x):
            for dy in range(-self.inflation_px, self.inflation_px + 1):
                for dx in range(-self.inflation_px, self.inflation_px + 1):
                    nx, ny = ox + dx, oy + dy
                    if 0 <= nx < self.map_w and 0 <= ny < self.map_h:
                        if dx*dx + dy*dy <= self.inflation_px*self.inflation_px:
                            blocked[ny][nx] = True
        return blocked

    def _is_collision_free(self, x1, y1, x2, y2):
        """
        Checks whether a straight line path drawn between two points (x1, y1 and x2, y2) passes through any blocked/inflated pixels
        Chops the line into tiny segments and tests each point against self.blocked. If even one point hits an obstacle, it returns False meaning collision
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

    def plan(self, start_xy, goal_xy, max_iter=3000, step_size=0.4, goal_bias=0.1):
        """
        Initializes the tree with the start_node and sets the loop up
        """
        start_node = Node(start_xy[0], start_xy[1])
        goal_node = Node(goal_xy[0], goal_xy[1])
        
        node_list = [start_node]

        for _ in range(max_iter):
            # 1. Picks a random coordinate in the area. With a 10% changce it purposefully picks the goal coordinate instead
            # keeps the tree growing toward the target instead of wandering randomly forever
            if random.random() < goal_bias:
                rnd_x, rnd_y = goal_node.x, goal_node.y
            else:
                rnd_x = random.uniform(self.origin_x, self.origin_x + self.map_w * self.res)
                rnd_y = random.uniform(self.origin_y, self.origin_y + self.map_h * self.res)

            # 2. (Nearest neighbor) - Scans all existing nodes in the tree to find the one closest to the newly sampled random point using
            #Euclidean distance (math.hypost)
            nearest_node = min(node_list, key=lambda n: math.hypot(n.x - rnd_x, n.y - rnd_y))

            # 3. (Steering) - Instead of jumping to the radom point a controlled step of length step_size from the nearest node towards random sample is taken 
            theta = math.atan2(rnd_y - nearest_node.y, rnd_x - nearest_node.x)
            dist = math.hypot(rnd_x - nearest_node.x, rnd_y - nearest_node.y)
            actual_step = min(step_size, dist)
            
            new_x = nearest_node.x + actual_step * math.cos(theta)
            new_y = nearest_node.y + actual_step * math.sin(theta)

            # 4. (Collision Check & Add): Calls ray tracer (_is_collision_free) to verify the path from the old node to the new node is clear. If it is clear it creates
            # a new node, links its parent to nearest node and adds it to the tree (node_list)
            if self._is_collision_free(nearest_node.x, nearest_node.y, new_x, new_y):
                new_node = Node(new_x, new_y)
                new_node.parent = nearest_node
                node_list.append(new_node)

                # 5. (Goal Check) - Checks if newly added node is close enough to the final goal.
                # If it is within step_size and has a clear line of sight to the goal, it snaps to the goal, 
                # attaches the parent link, and triggers path extraction
                if math.hypot(new_node.x - goal_node.x, new_node.y - goal_node.y) <= step_size:
                    if self._is_collision_free(new_node.x, new_node.y, goal_node.x, goal_node.y):
                        goal_node.parent = new_node
                        node_list.append(goal_node)
                        return self._extract_path(goal_node)

        return None  # Path not found within max_iter

    def _extract_path(self, node):
        """Starting from the goal node, it steps backwards using Current = current.parent 
        back to the start node collecting coordinates along the way
        """
        path = []
        current = node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]  # Reverses the list so the final output flows naturally from start to goal for the robot to follow.