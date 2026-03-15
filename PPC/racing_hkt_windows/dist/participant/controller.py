'''
PPC Hackathon — Participant Boilerplate
Implemented control() function
'''

import numpy as np

def steering(path: list[dict], state: dict) -> float:
    length_of_car = 2.6
    
    x, y, yaw, vx = state["x"], state["y"], state["yaw"], state["vx"]
    path_pts = np.array([[p["x"], p["y"]] for p in path])
    car_pos = np.array([x, y])

    # 1. Calculate dynamic lookahead distance (Ld) based on speed
    # Minimum 3.0m, increasing by 0.5m per 1m/s of speed
    ld = np.clip(0.5 * vx + 3.0, 3.0, 12.0) 

    # 2. Find the closest point on the path to the car
    distances = np.linalg.norm(path_pts - car_pos, axis=1)
    closest_idx = np.argmin(distances)

    # 3. Search forward to find the lookahead target point
    target_idx = closest_idx
    search_range = min(60, len(path_pts)) # Don't search the entire track to avoid wrapping backwards
    
    for i in range(search_range):
        idx = (closest_idx + i) % len(path_pts)
        dist = np.linalg.norm(path_pts[idx] - car_pos)
        if dist >= ld:
            target_idx = idx
            break

    target_pt = path_pts[target_idx]

    # 4. Calculate steering angle using Pure Pursuit mathematics
    dx = target_pt[0] - x
    dy = target_pt[1] - y

    # Angle to target in the world frame
    alpha = np.arctan2(dy, dx) - yaw
    
    # Normalize the angle to be within [-pi, pi]
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

    # Pure pursuit formula: steer = arctan(2 * L * sin(alpha) / Ld)
    actual_ld = np.linalg.norm([dx, dy])
    if actual_ld < 0.1: 
        actual_ld = 0.1 # Prevent division by zero
        
    steer = np.arctan2(2.0 * length_of_car * np.sin(alpha), actual_ld)

    # Clip steering to the maximum allowed angle of 0.5 radians (approx 28.6 degrees)
    return float(np.clip(steer, -0.5, 0.5))


def throttle_algorithm(target_speed: float, current_speed: float, dt: float) -> tuple[float, float]:
    # Initialize PID state variables as function attributes so they persist between calls
    if not hasattr(throttle_algorithm, 'integral'):
        throttle_algorithm.integral = 0.0
        throttle_algorithm.prev_error = 0.0

    # PID Tuning parameters
    Kp = 0.6  # Proportional gain
    Ki = 0.05 # Integral gain
    Kd = 0.1  # Derivative gain

    error = target_speed - current_speed
    
    # Anti-windup: only accumulate integral if we are not maxing out throttle/brake
    if -1.0 < error < 1.0:
        throttle_algorithm.integral += error * dt
        
    derivative = (error - throttle_algorithm.prev_error) / dt

    output = (Kp * error) + (Ki * throttle_algorithm.integral) + (Kd * derivative)
    throttle_algorithm.prev_error = error

    throttle = 0.0
    brake = 0.0

    if output > 0:
        throttle = output
    else:
        brake = -output

    # Clip throttle and brake to [0, 1]
    return float(np.clip(throttle, 0.0, 1.0)), float(np.clip(brake, 0.0, 1.0))


def control(
    path: list[dict],
    state: dict,
    cmd_feedback: dict,
    step: int,
) -> tuple[float, float, float]:
    """
    Generate throttle, steer, brake for the current timestep.
    Called every 50ms during simulation.
    """
    
    # 1. Calculate steering
    steer = steering(path, state)
    
    # 2. Calculate dynamic target speed based on steering angle
    # Go faster on straights (e.g., 8.0 m/s) and slow down on sharp turns (down to 4.0 m/s)
    base_speed = 8.0
    speed_penalty = (abs(steer) / 0.5) * 4.0 
    target_speed = base_speed - speed_penalty
    
    # 3. Calculate throttle and brake commands
    dt = 0.05 # 50ms time step as noted in boilerplate
    throttle, brake = throttle_algorithm(target_speed, state["vx"], dt)

    return throttle, steer, brake
