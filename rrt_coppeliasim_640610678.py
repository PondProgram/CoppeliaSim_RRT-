import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import random
import csv

class Node:
    def __init__(self, x, y, theta=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = None
        self.cost = 0.0  

class RRTStar:
    def __init__(self, occupancy_map, start, goal, 
                 max_iter=3000, step_size=10.0, search_radius=20,
                 safety_margin=0.1):
        self.occupancy_map = occupancy_map
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.safety_margin = safety_margin
        self.nodes = [self.start]
        self.goal_sample_rate = 0.1
        self.min_rand = 0
        self.max_rand_x = occupancy_map.shape[1]
        self.max_rand_y = occupancy_map.shape[0]
        self.best_goal_node = None
        self.best_goal_cost = float('inf')
        
    def is_position_valid(self, x, y):
        """Check if a position is valid and collision-free"""
        if x < 0 or x >= self.occupancy_map.shape[1] or y < 0 or y >= self.occupancy_map.shape[0]:
            return False
     
        if self.occupancy_map[int(y), int(x)] > self.safety_margin:
            return False
    
        robot_radius = 3  
        y_start = max(0, int(y - robot_radius))
        y_end = min(self.occupancy_map.shape[0], int(y + robot_radius + 1))
        x_start = max(0, int(x - robot_radius))
        x_end = min(self.occupancy_map.shape[1], int(x + robot_radius + 1))
    
        y_coords, x_coords = np.ogrid[y_start:y_end, x_start:x_end]
        dist_from_center = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        mask = dist_from_center <= robot_radius
     
        region = self.occupancy_map[y_start:y_end, x_start:x_end]
        return not np.any(region[mask] > self.safety_margin)

    def check_collision(self, from_node, to_point):
        """Check if path between two points is collision-free"""
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            return True
            
        steps = int(distance / (self.step_size/2))
        if steps == 0:
            steps = 1
            
        for i in range(steps + 1):
            x = from_node.x + dx * i / steps
            y = from_node.y + dy * i / steps
            if not self.is_position_valid(x, y):
                return False
        return True

    def get_random_point(self):
        """Get random point in configuration space"""
        if random.random() > self.goal_sample_rate:
            return (
                random.uniform(self.min_rand, self.max_rand_x),
                random.uniform(self.min_rand, self.max_rand_y)
            )
        return (self.goal.x, self.goal.y)

    def get_nearest_node(self, point):
        """Find nearest node in the tree"""
        distances = [(node.x - point[0])**2 + (node.y - point[1])**2 
                    for node in self.nodes]
        return self.nodes[distances.index(min(distances))]

    def steer(self, from_node, to_point):
        """Steer from one point toward another within step_size"""
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < self.step_size:
            new_x = to_point[0]
            new_y = to_point[1]
        else:
            theta = math.atan2(dy, dx)
            new_x = from_node.x + self.step_size * math.cos(theta)
            new_y = from_node.y + self.step_size * math.sin(theta)
            
        return (new_x, new_y)

    def get_neighbors(self, node, radius):
        """Find neighboring nodes within radius"""
        neighbors = []
        for potential_neighbor in self.nodes:
            if potential_neighbor == node:
                continue
            distance = math.sqrt((node.x - potential_neighbor.x)**2 + 
                               (node.y - potential_neighbor.y)**2)
            if distance <= radius:
                neighbors.append(potential_neighbor)
        return neighbors

    def choose_parent(self, new_node, neighbors):
        """Choose best parent for new node from neighbors"""
        if not neighbors:
            return
            
        costs = []
        for neighbor in neighbors:
            dx = new_node.x - neighbor.x
            dy = new_node.y - neighbor.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if self.check_collision(neighbor, (new_node.x, new_node.y)):
                costs.append(neighbor.cost + distance)
            else:
                costs.append(float('inf'))
                
        if not costs or min(costs) == float('inf'):
            return
            
        min_cost_idx = costs.index(min(costs))
        new_node.parent = neighbors[min_cost_idx]
        new_node.cost = costs[min_cost_idx]

    def rewire(self, new_node, neighbors):
        """Rewire the tree through new node if it provides better paths"""
        for neighbor in neighbors:
            dx = new_node.x - neighbor.x
            dy = new_node.y - neighbor.y
            distance = math.sqrt(dx**2 + dy**2)
            
            potential_cost = new_node.cost + distance
            
            if potential_cost < neighbor.cost and \
               self.check_collision(new_node, (neighbor.x, neighbor.y)):
                neighbor.parent = new_node
                neighbor.cost = potential_cost
                
                # Propagate cost updates to children
                self.update_children_cost(neighbor)
    
    def update_children_cost(self, node):
        """Update the cost of all children after rewiring"""
        for potential_child in self.nodes:
            if potential_child.parent == node:
                dx = potential_child.x - node.x
                dy = potential_child.y - node.y
                distance = math.sqrt(dx**2 + dy**2)
                potential_child.cost = node.cost + distance
                self.update_children_cost(potential_child)

    def calculate_angle_between_points(self, point1, point2):
        """Calculate angle between two points in degrees"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360

    def get_path(self):
        """Get the path from start to goal"""
        if self.goal.parent is None:
            return None
            
        path = [(self.goal.x, self.goal.y)]
        node = self.goal.parent
        
        while node != self.start:
            path.append((node.x, node.y))
            node = node.parent
            
        path.append((self.start.x, self.start.y))
        return path[::-1]

    def get_path_with_angles(self):
        """Get path with angle calculations between waypoints"""
        if self.goal.parent is None:
            return None
            
        path = self.get_path()
        if not path:
            return None
            
        # Calculate angles between consecutive points
        path_with_angles = []
        for i in range(len(path)-1):
            angle = self.calculate_angle_between_points(path[i], path[i+1])
            path_with_angles.append((path[i], angle))
        
        # Add final position with goal orientation
        path_with_angles.append((path[-1], math.degrees(self.goal.theta) % 360))
        
        return path_with_angles

    def plan(self):
        """Plan the path using RRT*"""
        for i in range(self.max_iter):
            # Get random point
            rnd_point = self.get_random_point()
            
            # Find nearest node
            nearest_node = self.get_nearest_node(rnd_point)
            
            # Steer towards the random point
            new_point = self.steer(nearest_node, rnd_point)
            
            # Check if the new point is collision-free
            if not self.check_collision(nearest_node, new_point):
                continue
                
            # Create new node
            new_node = Node(new_point[0], new_point[1])
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + \
                math.sqrt((new_point[0] - nearest_node.x)**2 + 
                         (new_point[1] - nearest_node.y)**2)
            
            # Find neighbors
            neighbors = self.get_neighbors(new_node, self.search_radius)
            
            # Choose parent
            self.choose_parent(new_node, neighbors)
            
            # Add node to tree
            self.nodes.append(new_node)
            
            # Rewire
            self.rewire(new_node, neighbors)
            
            # Try to connect to goal
            distance_to_goal = math.sqrt((new_node.x - self.goal.x)**2 + 
                                       (new_node.y - self.goal.y)**2)
            
            if distance_to_goal <= self.step_size and \
               self.check_collision(new_node, (self.goal.x, self.goal.y)):
                potential_cost = new_node.cost + distance_to_goal
                
                # Update best path if this one is better
                if potential_cost < self.best_goal_cost:
                    self.goal.parent = new_node
                    self.goal.cost = potential_cost
                    self.best_goal_cost = potential_cost
                    self.best_goal_node = new_node
            
            # Dynamic search radius based on number of nodes
            if len(self.nodes) > 100:
                self.search_radius = min(
                    50.0,  # Maximum radius
                    max(20.0,  # Minimum radius
                        50.0 * math.sqrt(math.log(len(self.nodes)) / len(self.nodes))
                    )
                )
                
            # Increase goal sampling rate as iterations progress
            self.goal_sample_rate = min(0.5, 0.1 + i / (self.max_iter * 2))
                
        # Return the best path found
        if self.best_goal_node is not None:
            self.goal.parent = self.best_goal_node
            return self.get_path()
        return None

    def plot_tree(self, path_with_angles=None):
        """Plot the RRT* tree and path"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.occupancy_map, cmap='gray_r', origin='lower')
        
        # Plot tree
        for node in self.nodes:
            if node.parent:
                plt.plot([node.x, node.parent.x], 
                        [node.y, node.parent.y], 'g-', alpha=0.3)
        
        # Plot path if found
        if path_with_angles:
            # Extract just the coordinates from path_with_angles
            path_coords = np.array([point for point, angle in path_with_angles])
            plt.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=2, label='Planned Path')
            
            # Plot waypoints with numbers and angles
            for i, (point, angle) in enumerate(path_with_angles):
                x, y = point
                plt.plot(x, y, 'bo', markersize=3)
                plt.annotate(f'{i+1}\n{angle:.1f}°', (x, y), 
                           textcoords="offset points", xytext=(0,10),
                           ha='center', fontsize=8)
        
        # Plot start position with orientation
        start_circle = Circle((self.start.x, self.start.y), 
                            radius=3, color='green', fill=True, alpha=0.7,
                            label='Start Position')
        plt.gca().add_patch(start_circle)
        
        # Add arrow for start orientation
        arrow_length = 5
        dx = arrow_length * np.cos(self.start.theta)
        dy = arrow_length * np.sin(self.start.theta)
        plt.arrow(self.start.x, self.start.y, dx, dy, 
                head_width=2, head_length=2, fc='green', ec='green')
        
        # Plot goal position with orientation
        goal_circle = Circle((self.goal.x, self.goal.y), 
                        radius=3, color='red', fill=True, alpha=0.7,
                        label='Goal Position')
        plt.gca().add_patch(goal_circle)
        
        # Add arrow for goal orientation
        dx = arrow_length * np.cos(self.goal.theta)
        dy = arrow_length * np.sin(self.goal.theta)
        plt.arrow(self.goal.x, self.goal.y, dx, dy, 
                head_width=2, head_length=2, fc='red', ec='red')
        
        plt.title('RRT* Path Planning with Waypoints and Orientations')
        plt.xlabel('X (grid cells)')
        plt.ylabel('Y (grid cells)')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

    def calculate_path_length(self, path_with_angles):
        """Calculate total path length"""
        total_length = 0
        for i in range(len(path_with_angles) - 1):
            (x1, y1), _ = path_with_angles[i]
            (x2, y2), _ = path_with_angles[i + 1]
            total_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return total_length
        
    def save_waypoints_to_csv(self, path_with_angles, filename='waypoints.csv'):
        """Save waypoints with additional path quality metrics"""
        def transform_coordinates(x, y):
            transformed_x = x * 0.02 - 2.5
            transformed_y = y * 0.02 - 2.5
            return transformed_x, transformed_y
            
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Transform and write start position
            start_x, start_y = transform_coordinates(self.start.x, self.start.y)
            writer.writerow([
                round(start_x, 1),
                round(start_y, 1),
                round(math.degrees(self.start.theta) % 360, 1),
            ])
            
            # Transform and write intermediate waypoints
            for i in range(1, len(path_with_angles) - 1):
                (x, y), angle = path_with_angles[i]
                transformed_x, transformed_y = transform_coordinates(x, y)
                writer.writerow([
                    round(transformed_x, 1),
                    round(transformed_y, 1),
                    round(angle, 1)
                ])
                
            # Transform and write goal position
            goal_x, goal_y = transform_coordinates(self.goal.x, self.goal.y)
            writer.writerow([
                round(goal_x, 1),
                round(goal_y, 1),
                round(math.degrees(self.goal.theta) % 360, 1),
            ])

def main():
    # Load map and get start/goal positions
    map_filename = 'ocgm_640610678.csv'
    occupancy_map = np.loadtxt(map_filename, delimiter=',')
    
    def random_valid_position(occupancy_map):
        # Define robot radius and safety margin
        robot_radius = 3  # Same as in RRTStar class
        safety_margin = 5  # Extra margin for safety
        total_margin = robot_radius + safety_margin
        
        # Get valid ranges considering margins
        y_max, x_max = occupancy_map.shape
        x_range = (total_margin, x_max - total_margin)
        y_range = (total_margin, y_max - total_margin)
        
        max_attempts = 1000
        attempt = 0
        
        while attempt < max_attempts:
            # Generate random position within valid ranges
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            
            # Convert to integers for array indexing
            x_int, y_int = int(x), int(y)
            
            # Check circular region around point
            y_start = max(0, y_int - total_margin)
            y_end = min(y_max, y_int + total_margin + 1)
            x_start = max(0, x_int - total_margin)
            x_end = min(x_max, x_int + total_margin + 1)
            
            # Create circular mask
            y_coords, x_coords = np.ogrid[y_start:y_end, x_start:x_end]
            dist_from_center = np.sqrt((x_coords - x_int)**2 + (y_coords - y_int)**2)
            mask = dist_from_center <= total_margin
            
            # Check if any obstacles in the region
            region = occupancy_map[y_start:y_end, x_start:x_end]
            if not np.any(region[mask] > 0.1):
                # Valid position found
                theta = np.random.uniform(0, 2 * np.pi)
                return (x_int, y_int, theta)
                
            attempt += 1
        
        raise RuntimeError("Could not find valid position after maximum attempts")
    
    start_pos = random_valid_position(occupancy_map)
    goal_pos = random_valid_position(occupancy_map)
    
    print("\nRRT* Path Planning")
    print("=================")
    print("\nInitial Configuration:")
    print("---------------------")
    print(f"Start position: (x: {start_pos[0]:.1f}, y: {start_pos[1]:.1f}, θ: {np.degrees(start_pos[2]):.1f}°)")
    print(f"Goal position: (x: {goal_pos[0]:.1f}, y: {goal_pos[1]:.1f}, θ: {np.degrees(goal_pos[2]):.1f}°)")
    
    # Create RRT* planner with optimized parameters
    rrt_star = RRTStar(
        occupancy_map, 
        start_pos, 
        goal_pos,
        max_iter=5000,  # Increased iterations for better optimization
        step_size=8.0,  # Smaller step size for finer paths
        search_radius=30.0,  # Larger initial search radius
        safety_margin=0.1  # Safety margin for obstacles
    )
    
    print("\nPlanning path...")
    # Plan path
    path = rrt_star.plan()
    
    if path:
        path_with_angles = rrt_star.get_path_with_angles()
        rrt_star.save_waypoints_to_csv(path_with_angles)
        
        print("\nPath Found Successfully!")
        print("=====================")
        print(f"Total waypoints: {len(path_with_angles)}")
        print(f"Path cost: {rrt_star.best_goal_cost:.2f}")
        print(f"Path length: {rrt_star.calculate_path_length(path_with_angles):.2f}")
        print(f"Number of nodes in tree: {len(rrt_star.nodes)}")
        print("\nWaypoint Details:")
        print("----------------")
        
        for i, ((x, y), angle) in enumerate(path_with_angles):
            print(f"\nWaypoint #{i+1}:")
            print(f"  Position: (x: {x:.1f}, y: {y:.1f})")
            print(f"  Heading: {angle:.1f}°")
            
            if i < len(path_with_angles) - 1:
                next_x, next_y = path_with_angles[i+1][0]
                distance = math.sqrt((next_x - x)**2 + (next_y - y)**2)
                print(f"  Distance to next: {distance:.1f} units")
        
        print("\nGenerating visualization...")
        rrt_star.plot_tree(path_with_angles)
        print("\nPath saved to 'waypoints.csv'")
        
    else:
        print("\nNo path found!")
        print("Planning tree visualized for debugging.")
        rrt_star.plot_tree()

if __name__ == "__main__":
    main()