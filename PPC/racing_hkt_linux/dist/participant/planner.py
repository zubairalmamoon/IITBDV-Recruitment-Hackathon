'''
PPC Hackathon — Participant Boilerplate
You must implement two functions: plan() and control()
'''

import numpy as np

def plan(cones: list[dict]) -> list[dict]:
    """
    Generate a highly accurate centerline by digitally densifying the boundaries.
    """
    path = []
    
    # 1. Sort cones to maintain track progression
    blue_cones = sorted([c for c in cones if c["side"] == "left"], key=lambda c: c["index"])
    yellow_cones = sorted([c for c in cones if c["side"] == "right"], key=lambda c: c["index"])
    
    if not blue_cones or not yellow_cones:
        return path

    B = np.array([[c["x"], c["y"]] for c in blue_cones])
    Y = np.array([[c["x"], c["y"]] for c in yellow_cones])

    # 2. Helper function to create a "solid wall" by interpolating points closely
    def create_dense_boundary(points, interval=0.1):
        # Calculate cumulative distance along the cones
        diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dists = np.concatenate(([0], np.cumsum(dists)))
        
        # Create an array of evenly spaced distances (every 10cm)
        even_dists = np.arange(0, cum_dists[-1], interval)
        
        # Interpolate X and Y coordinates along these distances
        x_dense = np.interp(even_dists, cum_dists, points[:, 0])
        y_dense = np.interp(even_dists, cum_dists, points[:, 1])
        return np.column_stack((x_dense, y_dense))

    # 3. Densify both boundaries to prevent chord-cutting
    B_dense = create_dense_boundary(B, interval=0.1)
    Y_dense = create_dense_boundary(Y, interval=0.1)

    # 4. Use the longer boundary as the primary reference to ensure full coverage on curves
    if len(B_dense) > len(Y_dense):
        primary = B_dense
        secondary = Y_dense
    else:
        primary = Y_dense
        secondary = B_dense

    waypoints = []
    
    # 5. Calculate strict geometric midpoints
    for p_pt in primary:
        # Find the absolute closest point on the solid opposite wall
        dists = np.linalg.norm(secondary - p_pt, axis=1)
        closest_sec = secondary[np.argmin(dists)]
        
        # Calculate true midpoint
        midpoint = (p_pt + closest_sec) / 2.0
        
        # Space out the final waypoints (e.g., every 0.5 meters) so the controller isn't overwhelmed
        if len(waypoints) == 0 or np.linalg.norm(waypoints[-1] - midpoint) >= 0.5:
            waypoints.append(midpoint)

    # 6. Format the output
    for pt in waypoints:
        path.append({"x": float(pt[0]), "y": float(pt[1])})

    return path
