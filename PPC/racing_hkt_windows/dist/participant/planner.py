'''
PPC Hackathon — Participant Boilerplate
You must implement two functions: plan() and control()
'''

# ─── TYPES (for reference) ────────────────────────────────────────────────────

# Cone: {"x": float, "y": float, "side": "left" | "right", "index": int}
# State: {"x", "y", "yaw", "vx", "vy", "yaw_rate"}  
# CmdFeedback: {"throttle", "steer"}        

# ─── PLANNER ──────────────────────────────────────────────────────────────────
import numpy as np

def plan(cones: list[dict]) -> list[dict]:
    """
    Generate a path from the cone layout.
    Called ONCE before the simulation starts.

    Args:
        cones: List of cone dicts with keys x, y, side ("left"/"right"), index

    Returns:
        path: List of waypoints [{"x": float, "y": float}, ...]
              Ordered from start to finish.
    """
    path = []
    
    # Sort cones by index to maintain track order
    blue_cones = sorted([c for c in cones if c["side"] == "left"], key=lambda c: c["index"])
    yellow_cones = [c for c in cones if c["side"] == "right"]
    
    if not blue_cones or not yellow_cones:
        return path

    yellow_coords = np.array([[c["x"], c["y"]] for c in yellow_cones])
    raw_waypoints = []

    # Find the nearest right cone for every left cone to calculate track midpoints
    for b_cone in blue_cones:
        b_pos = np.array([b_cone["x"], b_cone["y"]])
        
        # Calculate Euclidean distance to all yellow cones
        distances = np.linalg.norm(yellow_coords - b_pos, axis=1)
        closest_y_idx = np.argmin(distances)
        closest_y_pos = yellow_coords[closest_y_idx]
        
        # Calculate midpoint
        midpoint = (b_pos + closest_y_pos) / 2.0
        raw_waypoints.append(midpoint)
        
    raw_waypoints = np.array(raw_waypoints)

    # Smooth the path using a moving average window to create a better racing line
    window_size = 5
    if len(raw_waypoints) >= window_size:
        kernel = np.ones(window_size) / window_size
        smooth_x = np.convolve(raw_waypoints[:, 0], kernel, mode='same')
        smooth_y = np.convolve(raw_waypoints[:, 1], kernel, mode='same')
        
        # Handle edges affected by 'same' convolution by keeping raw values at the very ends
        pad = window_size // 2
        smooth_x[:pad] = raw_waypoints[:pad, 0]
        smooth_x[-pad:] = raw_waypoints[-pad:, 0]
        smooth_y[:pad] = raw_waypoints[:pad, 1]
        smooth_y[-pad:] = raw_waypoints[-pad:, 1]
    else:
        smooth_x = raw_waypoints[:, 0]
        smooth_y = raw_waypoints[:, 1]

    # Format output
    for x, y in zip(smooth_x, smooth_y):
        path.append({"x": float(x), "y": float(y)})

    return path
