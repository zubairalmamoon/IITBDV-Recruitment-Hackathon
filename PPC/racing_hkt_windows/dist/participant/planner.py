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
    
    # Sort cones by index to maintain general track order
    blue_cones = sorted([c for c in cones if c["side"] == "left"], key=lambda c: c["index"])
    yellow_cones = sorted([c for c in cones if c["side"] == "right"], key=lambda c: c["index"])
    
    if not blue_cones or not yellow_cones:
        return path

    B = np.array([[c["x"], c["y"]] for c in blue_cones])
    Y = np.array([[c["x"], c["y"]] for c in yellow_cones])

    waypoints = []

    # Iterate through the boundary that has MORE cones to ensure we don't cut corners
    if len(B) >= len(Y):
        primary = B
        secondary = Y
    else:
        primary = Y
        secondary = B

    for pt in primary:
        # 1. Find the strictly closest cone on the opposite side
        dists = np.linalg.norm(secondary - pt, axis=1)
        closest_pt = secondary[np.argmin(dists)]
        
        # 2. Calculate true geometric midpoint
        midpoint = (pt + closest_pt) / 2.0
        
        # Prevent duplicate waypoints if multiple outer cones map to the same inner apex cone
        if len(waypoints) == 0 or np.linalg.norm(waypoints[-1] - midpoint) > 0.5:
            waypoints.append(midpoint)

    # 3. Format output directly. 
    # NO SMOOTHING APPLIED. Smoothing shrinks the turn radius and causes cone strikes.
    for pt in waypoints:
        path.append({"x": float(pt[0]), "y": float(pt[1])})

    return path
