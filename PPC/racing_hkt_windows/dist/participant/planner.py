'''
PPC Hackathon — Participant Boilerplate
You must implement two functions: plan() and control()
'''

import numpy as np

def plan(cones: list[dict]) -> list[dict]:
    """
    Generate a path from the cone layout using the mean of matched cones.
    Called ONCE before the simulation starts.
    """
    path = []
    
    # 1. Sort cones by their provided index to ensure they are in track order
    blue_cones = sorted([c for c in cones if c["side"] == "left"], key=lambda c: c["index"])
    yellow_cones = sorted([c for c in cones if c["side"] == "right"], key=lambda c: c["index"])
    
    if not blue_cones or not yellow_cones:
        return path

    # 2. Find the minimum length in case there are missing cones on one side
    min_length = min(len(blue_cones), len(yellow_cones))
    
    raw_waypoints = []

    # 3. Calculate the mean (midpoint) for each matching pair of cones
    for i in range(min_length):
        b_x = blue_cones[i]["x"]
        b_y = blue_cones[i]["y"]
        
        y_x = yellow_cones[i]["x"]
        y_y = yellow_cones[i]["y"]
        
        mid_x = (b_x + y_x) / 2.0
        mid_y = (b_y + y_y) / 2.0
        
        raw_waypoints.append([mid_x, mid_y])

    raw_waypoints = np.array(raw_waypoints)

    # 4. Optional: Lightly smooth the path to prevent sudden steering jerks
    # (A small window size of 3 just takes the edge off any misplaced cones)
    window_size = 3
    if len(raw_waypoints) >= window_size:
        kernel = np.ones(window_size) / window_size
        smooth_x = np.convolve(raw_waypoints[:, 0], kernel, mode='same')
        smooth_y = np.convolve(raw_waypoints[:, 1], kernel, mode='same')
        
        # Keep the exact start and end points
        smooth_x[0] = raw_waypoints[0, 0]
        smooth_x[-1] = raw_waypoints[-1, 0]
        smooth_y[0] = raw_waypoints[0, 1]
        smooth_y[-1] = raw_waypoints[-1, 1]
    else:
        smooth_x = raw_waypoints[:, 0]
        smooth_y = raw_waypoints[:, 1]

    # 5. Format the output to match the required list of dictionaries
    for x, y in zip(smooth_x, smooth_y):
        path.append({"x": float(x), "y": float(y)})

    return path
