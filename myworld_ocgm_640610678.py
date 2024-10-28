import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import binary_dilation, binary_erosion

# Grid settings
grid_size = 250  
grid_resolution = 0.02  

# Grid initialization
occupancy_grid = np.ones((grid_size, grid_size)) * 0.5
detection_history = np.zeros((grid_size, grid_size))

# Probability settings
P_OCC = 0.95
P_FREE = 0.05

# Modified dilation settings - increased kernel size for larger borders
dilation_kernel = np.ones((7, 7))  # Increased from (5, 5)
obstacle_threshold = 0.65  # Slightly lowered from 0.7 to catch more potential obstacles

def binarize_and_dilate_map(occupancy_data):
    """
    Convert map to binary (black-white) and perform dilation.
    Areas without data (-1) will be converted to free space (0).
    Enhanced dilation for thicker borders.
    """
    # Create binary map starting with all white (0)
    binary_map = np.zeros_like(occupancy_data)
    
    # Set to black (100) only areas that are definite obstacles
    binary_map[occupancy_data >= obstacle_threshold] = 100
    
    # Perform multiple dilations for thicker borders
    dilated_map = binary_dilation(binary_map == 100, structure=dilation_kernel, iterations=2)
    
    # Convert back to occupancy values (0-100)
    result_map = np.zeros_like(occupancy_data)
    result_map[dilated_map] = 100
    
    return result_map

def bresenham(x0, y0, x1, y1):
    """Bresenham's line algorithm (vectorized)"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    points = []
    x, y = x0, y0
    err = dx - dy
    
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
            
    return np.array(points)

def update_cell(x, y, is_occupied, strength=1.0):
    """Update cell probability and detection history"""
    if 0 <= x < grid_size and 0 <= y < grid_size:
        current_prob = occupancy_grid[y, x]
        
        if is_occupied:
            detection_history[y, x] += 1
            new_prob = current_prob + (P_OCC - current_prob) * strength
        else:
            if detection_history[y, x] < 2:
                new_prob = current_prob + (P_FREE - current_prob) * strength * 0.5
            else:
                new_prob = current_prob
            
        occupancy_grid[y, x] = np.clip(new_prob, 0.01, 0.99)

def enhance_edges(grid, history):
    """Enhanced edge detection with stronger borders"""
    enhanced = grid.copy()
    
    # Stronger edge enhancement
    first_detection = (history == 1) & (grid > 0.65)  # Lowered threshold
    enhanced[first_detection] = 0.7  # Increased from 0.6
    
    confirmed = (history >= 2) & (grid > 0.65)
    enhanced[confirmed] = 0.98  # Increased from 0.95
    
    # Apply enhanced dilation to confirmed obstacles
    dilated = binarize_and_dilate_map(enhanced)
    
    # Stronger blending of dilated map
    enhanced = np.maximum(enhanced, dilated / 100 * 0.9)  # Increased from 0.8
    
    return enhanced

def update_occupancy_grid(x_robot, y_robot, th_robot, sensor_lengths, sensor_angles):
    robot_x_grid = int(x_robot / grid_resolution) + grid_size // 2
    robot_y_grid = int(y_robot / grid_resolution) + grid_size // 2
    
    for i in range(len(sensor_lengths)):
        if np.isnan(sensor_lengths[i]) or np.isinf(sensor_lengths[i]):
            continue
            
        max_range = 1.9  
        
        if sensor_lengths[i] >= max_range:
            laser_x = x_robot + max_range * np.cos(th_robot + sensor_angles[i])
            laser_y = y_robot + max_range * np.sin(th_robot + sensor_angles[i])
            laser_x_grid = int(laser_x / grid_resolution) + grid_size // 2
            laser_y_grid = int(laser_y / grid_resolution) + grid_size // 2
            
            if not (0 <= laser_x_grid < grid_size and 0 <= laser_y_grid < grid_size):
                continue
                
            line_points = bresenham(robot_x_grid, robot_y_grid, laser_x_grid, laser_y_grid)
            
            if len(line_points) > 0:
                for x, y in line_points:
                    update_cell(x, y, False, strength=0.1)
        else:
            laser_x = x_robot + sensor_lengths[i] * np.cos(th_robot + sensor_angles[i])
            laser_y = y_robot + sensor_lengths[i] * np.sin(th_robot + sensor_angles[i])
            laser_x_grid = int(laser_x / grid_resolution) + grid_size // 2
            laser_y_grid = int(laser_y / grid_resolution) + grid_size // 2
            
            if not (0 <= laser_x_grid < grid_size and 0 <= laser_y_grid < grid_size):
                continue
                
            line_points = bresenham(robot_x_grid, robot_y_grid, laser_x_grid, laser_y_grid)
            
            if len(line_points) > 1:
                for x, y in line_points[:-1]:
                    update_cell(x, y, False, strength=0.1)
            
            if len(line_points) > 0:
                x, y = line_points[-1]
                update_cell(x, y, True, strength=0.85)  # Increased from 0.8

def save_occupancy_grid(filename):
    final_grid = occupancy_grid.copy()
    
    # Enhanced obstacle confirmation
    obstacle_confirmed = (detection_history >= 2) & (occupancy_grid > 0.65)  # Lowered from 0.7
    final_grid[obstacle_confirmed] = 1.0
    free_confirmed = (detection_history >= 2) & (occupancy_grid < 0.3)
    final_grid[free_confirmed] = 0.0
    
    # Apply enhanced dilation to final map
    final_grid = binarize_and_dilate_map(final_grid) / 100.0
    
    np.save(filename.replace('.csv', '.npy'), final_grid)
    np.savetxt(filename, final_grid, delimiter=',', fmt='%.3f')
    
    print(f"Saved occupancy grid to {filename} and {filename.replace('.csv', '.npy')}")

def update(frame):
    update_occupancy_grid(X[frame], Y[frame], TH[frame], 
                         sensor_length[frame], sensor_angle[frame])
    
    ax.clear()
    
    enhanced_grid = enhance_edges(occupancy_grid, detection_history)
    ax.imshow(enhanced_grid, cmap='gray_r', origin='lower', vmin=0, vmax=1)
    
    robot_x_grid = int(X[frame] / grid_resolution) + grid_size // 2
    robot_y_grid = int(Y[frame] / grid_resolution) + grid_size // 2
    ax.plot(robot_x_grid, robot_y_grid, 'ro', markersize=5)  
    
    direction_length = 10
    dx = direction_length * np.cos(TH[frame])
    dy = direction_length * np.sin(TH[frame])
    ax.plot([robot_x_grid, robot_x_grid + dx], 
            [robot_y_grid, robot_y_grid + dy], 'r-', linewidth=2)
    
    ax.set_title(f'Frame {frame+1}/{len(X)}')
    ax.set_xlim([0, grid_size])
    ax.set_ylim([0, grid_size])
    ax.set_aspect('equal')
    
    if frame == len(X) - 1:
        save_occupancy_grid('ocgm_640610678.csv')
    
    return ax,

# Load and prepare data
data = np.loadtxt('data.csv', delimiter=',')
X = data[:, 0] 
Y = data[:, 1]  
TH = data[:, 2]  

num_ray = int((data.shape[1] - 3) / 2)
sensor_length = np.empty((data.shape[0], num_ray))
sensor_angle = np.empty((data.shape[0], num_ray))

for i in range(num_ray):
    sensor_length[:, i] = data[:, 3 + 2 * i]  
    sensor_angle[:, i] = data[:, 4 + 2 * i]  

fig, ax = plt.subplots(figsize=(8, 8))
ani = FuncAnimation(fig, update, frames=len(X), interval=50, repeat=False, blit=True)
plt.show()