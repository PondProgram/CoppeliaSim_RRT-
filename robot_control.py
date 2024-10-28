import math
import time
import csv
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class MecanumRobotController:
    def __init__(self):
        # Initialize connection to CoppeliaSim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        
        # Robot parameters
        self.wheel_radius = 0.06  # meters
        self.robot_width = 0.213    # meters
        self.robot_length = 0.234   # meters
        self.max_wheel_velocity = 3.0  # rad/s
        
        # Control parameters
        self.pos_tolerance = 0.05    # meters
        self.angle_tolerance = 0.1   # radians
        self.Kp_linear = 0.9        # Proportional gain for linear motion
        self.Kp_angular = 1.1       # Proportional gain for angular motion
        self.linear_velocity = 0.25  # m/s
        self.angular_velocity = 0.35 # rad/s
        
        # Initialize robot handles
        self.init_robot_handles()

    def init_robot_handles(self):
        """Initialize handles for robot and wheels"""
        # Get robot base handle
        self.robot_handle = self.sim.getObject('/turbo')
        
        # Get wheel handles
        self.wheel_handles = {
            'front_left': self.sim.getObject('/turbo/wheel_FL_1_joint'),
            'front_right': self.sim.getObject('/turbo/wheel_FR_1_joint'),
            'rear_left': self.sim.getObject('/turbo/wheel_RL_1_joint'),
            'rear_right': self.sim.getObject('/turbo/wheel_RR_1_joint')
        }

    def initialize_robot_position(self, x, y, theta):
        """Initialize robot at the specified position and orientation"""
        position = [x, y, 0]  # z-coordinate set to 0
        orientation = [0, 0, theta]  # Only yaw angle is set, pitch and roll are 0
        
        # Set robot position and orientation
        self.sim.setObjectPosition(self.robot_handle, -1, position)
        self.sim.setObjectOrientation(self.robot_handle, -1, orientation)
        
        print(f"Robot initialized at position: ({x:.2f}, {y:.2f}) and angle: {math.degrees(theta):.2f}°")
        time.sleep(0.5)  # Wait for physics to stabilize

    def get_robot_pose(self):
        """Get current robot position and orientation"""
        position = self.sim.getObjectPosition(self.robot_handle, -1)
        orientation = self.sim.getObjectOrientation(self.robot_handle, -1)
        return position, orientation[2]  # Return [x, y, z], yaw

    def set_wheel_velocities(self, vx, vy, omega):
        """
        Set wheel velocities for mecanum drive
        vx: forward velocity
        vy: lateral velocity 
        omega: angular velocity
        """
        # Mecanum wheel velocity calculations
        v_fl = vx - vy - (self.robot_width + self.robot_length) * omega
        v_fr = vx + vy + (self.robot_width + self.robot_length) * omega
        v_rl = vx + vy - (self.robot_width + self.robot_length) * omega
        v_rr = vx - vy + (self.robot_width + self.robot_length) * omega
        
        # Convert to angular velocities and apply limits
        wheel_velocities = {
            'front_left': np.clip(v_fl / self.wheel_radius, -self.max_wheel_velocity, self.max_wheel_velocity),
            'front_right': np.clip(v_fr / self.wheel_radius, -self.max_wheel_velocity, self.max_wheel_velocity),
            'rear_left': np.clip(v_rl / self.wheel_radius, -self.max_wheel_velocity, self.max_wheel_velocity),
            'rear_right': np.clip(v_rr / self.wheel_radius, -self.max_wheel_velocity, self.max_wheel_velocity)
        }
        
        # Set wheel velocities
        for wheel, velocity in wheel_velocities.items():
            self.sim.setJointTargetVelocity(self.wheel_handles[wheel], velocity)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return ((angle + math.pi) % (2 * math.pi)) - math.pi

    def rotate_to_angle(self, target_angle):
        """Rotate the robot to face a specific angle"""
        print(f"Rotating to angle: {math.degrees(target_angle):.2f}°")
        
        while True:
            # Get current orientation
            _, current_angle = self.get_robot_pose()
            
            # Calculate angle error
            angle_error = self.normalize_angle(target_angle - current_angle)
            
            # Check if target angle reached
            if abs(angle_error) < self.angle_tolerance:
                self.set_wheel_velocities(0, 0, 0)
                print("Target angle reached")
                return True
            
            # Calculate rotation direction and apply rotation
            angular_velocity = np.sign(angle_error) * self.angular_velocity
            self.set_wheel_velocities(0, 0, angular_velocity)
            time.sleep(0.01)

    def move_forward(self, target_x, target_y):
        """Move the robot forward in its current orientation"""
        print(f"Moving forward to position: ({target_x:.2f}, {target_y:.2f})")
        
        while True:
            # Get current position
            position, current_angle = self.get_robot_pose()
            current_x, current_y = position[0], position[1]
            
            # Calculate distance error
            dx = target_x - current_x
            dy = target_y - current_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Check if target reached
            if distance < self.pos_tolerance:
                self.set_wheel_velocities(0, 0, 0)
                print("Target position reached")
                return True
            
            # Move forward in robot's current orientation
            speed = min(self.linear_velocity, self.Kp_linear * distance)
            self.set_wheel_velocities(speed, 0, 0)
            time.sleep(0.01)

    def move_to_position(self, target_x, target_y, target_theta):
        """Move to position with point-and-move strategy"""
        print(f"\nMoving to target: ({target_x:.2f}, {target_y:.2f}, {math.degrees(target_theta):.2f}°)")
        
        # Get current position
        current_pos, _ = self.get_robot_pose()
        current_x, current_y = current_pos[0], current_pos[1]
        
        # Calculate angle to target
        dx = target_x - current_x
        dy = target_y - current_y
        angle_to_target = math.atan2(dy, dx)
        
        # First rotate to face the target
        print("Step 1: Rotating to face target...")
        self.rotate_to_angle(angle_to_target)
        
        # Then move forward to target
        print("Step 2: Moving forward to target...")
        self.move_forward(target_x, target_y)
        
        # Finally rotate to desired final orientation
        print("Step 3: Rotating to final orientation...")
        self.rotate_to_angle(target_theta)
        
        print("Target position and orientation reached")
        return True

    def follow_waypoints(self, waypoints_file):
        """Follow waypoints from CSV file with initialization"""
        # Read waypoints
        waypoints = []
        with open(waypoints_file, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                x, y, theta = map(float, row)
                theta = math.radians(theta)  # Convert to radians
                waypoints.append((x, y, theta))
        
        if not waypoints:
            print("No waypoints found!")
            return
        
        # Initialize robot at the first waypoint
        start_x, start_y, start_theta = waypoints[0]
        print("\nInitializing robot at starting position...")
        self.initialize_robot_position(start_x, start_y, start_theta)
        print("Robot initialized successfully")
        
        print("\nStarting waypoint navigation...")
        print(f"Total waypoints: {len(waypoints)}")
        
        # Move through remaining waypoints
        for i, (x, y, theta) in enumerate(waypoints[1:], start=1):
            print(f"\nMoving to waypoint {i}/{len(waypoints)-1}")
            print(f"Target: x={x:.2f}, y={y:.2f}, theta={math.degrees(theta):.2f}°")
            self.move_to_position(x, y, theta)
            print(f"Waypoint {i} reached")
            time.sleep(0.5)  # Short pause between waypoints
        
        print("\nNavigation completed!")

    def stop(self):
        """Stop all wheel movements"""
        self.set_wheel_velocities(0, 0, 0)

def main():
    # Create controller
    controller = MecanumRobotController()
    
    # Start simulation
    controller.sim.startSimulation()
    time.sleep(0.5)  # Wait for simulation to stabilize
    
    try:
        # Follow waypoints (includes initialization at first waypoint)
        controller.follow_waypoints('waypoints.csv')
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Stop robot and simulation
        controller.stop()
        controller.sim.stopSimulation()

if __name__ == "__main__":
    main()