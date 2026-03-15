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

    # 2. Function to calculate normalized cumulative distance along a boundary
    def calc_normalized_dist(points):
        # Calculate distances between consecutive points
        diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        # Cumulative sum of distances
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        # Normalize to range [0, 1]
        if cum_dist[-1] > 0:
            cum_dist = cum_dist / cum_dist[-1]
        return cum_dist

    # Get the normalized progress [0.0 to 1.0] for both boundaries
    t_B = calc_normalized_dist(B)
    t_Y = calc_normalized_dist(Y)
    
    # 3. Create a common progress vector with high resolution
    # Using twice the max number of cones ensures a very smooth, dense path
    num_points = max(len(B), len(Y)) * 2
    t_common = np.linspace(0, 1, num_points)
    
    # 4. Interpolate X and Y coordinates for both boundaries at the common progress points
    Bx_interp = np.interp(t_common, t_B, B[:, 0])
    By_interp = np.interp(t_common, t_B, B[:, 1])
    
    Yx_interp = np.interp(t_common, t_Y, Y[:, 0])
    Yy_interp = np.interp(t_common, t_Y, Y[:, 1])
    
    # 5. Calculate true midpoints
    mid_x = (Bx_interp + Yx_interp) / 2.0
    mid_y = (By_interp + Yy_interp) / 2.0
    
    # Format output
    for x, y in zip(mid_x, mid_y):
        path.append({"x": float(x), "y": float(y)})

    return path
