'''
PPC Hackathon — Participant Boilerplate
You must implement two functions: plan() and control()
'''

import numpy as np

def steering(path: list[dict], state: dict) -> float:
    length_of_car = 2.6
    x, y, yaw, vx = state["x"], state["y"], state["yaw"], state["vx"]
    
    path_pts = np.array([[p["x"], p["y"]] for p in path])
    car_pos = np.array([x, y])

    # 1. ULTRA-TIGHT Lookahead Distance (Ld)
    # Base of 1.5m, slightly increasing with speed. Maxes out at 4.5m.
    # This forces the car to strictly follow the dots right in front of its nose.
    ld = np.clip(0.25 * vx + 1.5, 1.5, 4.5) 

    distances = np.linalg.norm(path_pts - car_pos, axis=1)
    closest_idx = np.argmin(distances)

    target_idx = closest_idx
    search_range = min(50, len(path_pts)) 
    
    for i in range(search_range):
        idx = (closest_idx + i) % len(path_pts)
        dist = np.linalg.norm(path_pts[idx] - car_pos)
        if dist >= ld:
            target_idx = idx
            break

    target_pt = path_pts[target_idx]

    dx = target_pt[0] - x
    dy = target_pt[1] - y

    alpha = np.arctan2(dy, dx) - yaw
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

    actual_ld = np.linalg.norm([dx, dy])
    if actual_ld < 0.1: 
        actual_ld = 0.1 
        
    steer = np.arctan2(2.0 * length_of_car * np.sin(alpha), actual_ld)

    # 0.5 rad is the max steering angle
    return float(np.clip(steer, -0.5, 0.5))


def throttle_algorithm(target_speed: float, current_speed: float, dt: float) -> tuple[float, float]:
    # Simplified PD Controller (Removed Integral to stop the aggressive vibrating/barcode effect)
    if not hasattr(throttle_algorithm, 'prev_error'):
        throttle_algorithm.prev_error = 0.0

    Kp = 0.8  # Strong proportional response
    Kd = 0.1  # Dampens sudden changes

    error = target_speed - current_speed
    derivative = (error - throttle_algorithm.prev_error) / dt

    output = (Kp * error) + (Kd * derivative)
    throttle_algorithm.prev_error = error

    throttle = 0.0
    brake = 0.0

    if output > 0:
        throttle = output
    else:
        brake = -output

    return float(np.clip(throttle, 0.0, 1.0)), float(np.clip(brake, 0.0, 1.0))


def control(
    path: list[dict],
    state: dict,
    cmd_feedback: dict,
    step: int,
) -> tuple[float, float, float]:
    
    steer = steering(path, state)
    
    # 2. CONSERVATIVE Speed Profile
    # Slow down to guarantee we don't slide into cones
    base_speed = 5.0  # Max speed on straights
    
    # If the steering wheel is turned, aggressively drop the target speed
    speed_penalty = (abs(steer) / 0.5) * 3.0 
    
    # Minimum speed of 2.0 m/s around tight hairpins
    target_speed = np.clip(base_speed -
