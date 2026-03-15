'''
PPC Hackathon — Participant Boilerplate
You must implement two functions: plan() and control()
'''

import numpy as np

def plan(cones: list[dict]) -> list[dict]:
    """
    Generate a path from the cone layout.
    Called ONCE before the simulation starts.
    """
    path = []
    
    # 1. Sort cones by index to maintain track order
    blue_cones = sorted([c for c in cones if c["side"] == "left"], key=lambda c: c["index"])
    yellow_cones = sorted([c for c in cones if c["side"] == "right"], key=lambda c: c["index"])
    
    if not blue_cones or not yellow_cones:
        return path

    B = np.array([[c["x"], c["y"]] for c in blue_cones])
    Y = np.array([[c["x"], c["y"]] for c in yellow_cones])

    # 2. Function to create a highly dense line of points between the cones
    def densify(points, num_points=500):
        # Calculate cumulative distances between cones
        diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dists = np.concatenate(([0], np.cumsum(dists)))
        
        # Create an evenly spaced array of distances
        even_dists = np.linspace(0, cum_dists[-1], num_points)
        
        # Interpolate x and y coordinates to fill in the gaps between cones
        x_dense = np.interp(even_dists, cum_dists, points[:, 0])
        y_dense = np.interp(even_dists, cum_dists, points[:, 1])
        return np.column_stack((x_dense, y_dense))

    # Create continuous, dense boundaries
    B_dense = densify(B, 500)
    Y_dense = densify(Y, 500)

    raw_waypoints = []
    
    # 3. Geometric Matching: Find the strictly closest yellow point for every blue point
    for b_pt in B_dense:
        # Calculate distances from this specific blue point to ALL yellow points
        dists = np.linalg.norm(Y_dense - b_pt, axis=1)
        closest_y_idx = np.argmin(dists)
        closest_y_pt = Y_dense[closest_y_idx]
        
        # Calculate the true geometric midpoint across the track
        midpoint = (b_pt + closest_y_pt) / 2.0
        raw_waypoints.append(midpoint)
        
    raw_waypoints = np.array(raw_waypoints)

    # 4. Smooth the midpoints to ensure a drivable racing line without jagged steering
    window_size = 15
    if len(raw_waypoints) >= window_size:
        kernel = np.ones(window_size) / window_size
        smooth_x = np.convolve(raw_waypoints[:, 0], kernel, mode='same')
        smooth_y = np.convolve(raw_waypoints[:, 1], kernel, mode='same')
        
        # Protect the start and end points from convolution distortion
        pad = window_size // 2
        smooth_x[:pad] = raw_waypoints[:pad, 0]
        smooth_x[-pad:] = raw_waypoints[-pad:, 0]
        smooth_y[:pad] = raw_waypoints[:pad, 1]
        smooth_y[-pad:] = raw_waypoints[-pad:, 1]
    else:
        smooth_x = raw_waypoints[:, 0]
        smooth_y = raw_waypoints[:, 1]

    # 5. Format output (downsample slightly so we don't overwhelm the controller)
    step = max(1, len(smooth_x) // 150)
    for x, y in zip(smooth_x[::step], smooth_y[::step]):
        path.append({"x": float(x), "y": float(y)})

    return path
