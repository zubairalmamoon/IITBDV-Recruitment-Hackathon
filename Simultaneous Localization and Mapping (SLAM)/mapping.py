import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance
import pandas as pd

# ── Load Track from CSV ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_HERE, "small_track.csv"))

BLUE_CONES   = df[df["tag"] == "blue"      ][["x", "y"]].values.astype(float)
YELLOW_CONES = df[df["tag"] == "yellow"    ][["x", "y"]].values.astype(float)
BIG_ORANGE   = df[df["tag"] == "big_orange"][["x", "y"]].values.astype(float)

_cs               = df[df["tag"] == "car_start"].iloc[0]
CAR_START_POS     = np.array([float(_cs["x"]), float(_cs["y"])])
CAR_START_HEADING = float(_cs["direction"])   

MAP_CONES = np.vstack([BLUE_CONES, YELLOW_CONES])


# ── Build Approximate Centerline ──────────────────────────────────────────────
def _build_centerline():
    center = np.mean(MAP_CONES, axis=0)
    D      = distance.cdist(BLUE_CONES, YELLOW_CONES)
    mids   = np.array(
        [(BLUE_CONES[i] + YELLOW_CONES[np.argmin(D[i])]) / 2.0
         for i in range(len(BLUE_CONES))]
    )
    angles = np.arctan2(mids[:, 1] - center[1], mids[:, 0] - center[0])
    return mids[np.argsort(angles)[::-1]]

CENTERLINE = _build_centerline()


# ── Simulation Parameters ─────────────────────────────────────────────────────
SENSOR_RANGE = 12.0   
NOISE_STD    = 0.20   
WHEELBASE    = 3.0    
DT           = 0.1    
SPEED        = 7.0    
LOOKAHEAD    = 5.5    
N_FRAMES     = 130    


# ── Utility Functions ─────────────────────────────────────────────────────────
def angle_wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

def pure_pursuit(pos: np.ndarray, heading: float, path: np.ndarray) -> float:
    dists   = np.linalg.norm(path - pos, axis=1)
    nearest = int(np.argmin(dists))
    n       = len(path)
    target  = path[(nearest + 5) % n]       
    for k in range(nearest, nearest + n):
        pt = path[k % n]
        if np.linalg.norm(pt - pos) >= LOOKAHEAD:
            target = pt
            break
    alpha = angle_wrap(
        np.arctan2(target[1] - pos[1], target[0] - pos[0]) - heading
    )
    steer = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), LOOKAHEAD)
    return float(np.clip(steer, -0.6, 0.6))

def local_to_global(local_pts: np.ndarray, pos: np.ndarray, heading: float) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, -s], [s, c]])       
    return (R @ local_pts.T).T + pos

def get_measurements(pos: np.ndarray, heading: float) -> np.ndarray:
    dists   = np.linalg.norm(MAP_CONES - pos, axis=1)
    visible = MAP_CONES[dists < SENSOR_RANGE]
    if len(visible) == 0:
        return np.zeros((0, 2))
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, s], [-s, c]])       
    local = (R @ (visible - pos).T).T
    return local + np.random.normal(0, NOISE_STD, local.shape)

def step_kinematic(pos: np.ndarray, heading: float, velocity: float, steering: float):
    new_pos = pos.copy()
    new_pos[0] += velocity * np.cos(heading) * DT
    new_pos[1] += velocity * np.sin(heading) * DT
    new_heading = angle_wrap(
        heading + (velocity / WHEELBASE) * np.tan(steering) * DT
    )
    return new_pos, new_heading

def draw_track(ax, alpha_b: float = 0.4, alpha_y: float = 0.4) -> None:
    ax.scatter(BLUE_CONES[:, 0],   BLUE_CONES[:, 1],
               c="royalblue", marker="^", s=65,  alpha=alpha_b, zorder=2)
    ax.scatter(YELLOW_CONES[:, 0], YELLOW_CONES[:, 1],
               c="gold",      marker="^", s=65,  alpha=alpha_y, zorder=2)
    ax.scatter(BIG_ORANGE[:, 0],   BIG_ORANGE[:, 1],
               c="darkorange", marker="s", s=100, alpha=0.7, zorder=2)

def draw_car(ax, pos: np.ndarray, heading: float) -> None:
    ax.scatter(pos[0], pos[1], c="red", s=160, zorder=7)
    ax.arrow(pos[0], pos[1],
             2.2 * np.cos(heading), 2.2 * np.sin(heading),
             head_width=0.8, fc="red", ec="red", zorder=8)

def setup_ax(ax, subtitle: str = "") -> None:
    ax.set_xlim(-28, 28)
    ax.set_ylim(-22, 22)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    if subtitle:
        ax.set_title(subtitle, fontsize=10)


# ── Abstract Base ─────────────────────────────────────────────────────────────
class Bot:
    def __init__(self):
        self.pos     = CAR_START_POS.copy()   
        self.heading = CAR_START_HEADING      

    def data_association(self, measurements, current_map):
        raise NotImplementedError

    def localization(self, velocity, steering):
        raise NotImplementedError

    def mapping(self, measurements):
        raise NotImplementedError


# ──  Solution ──────────────────────────────────────────────────────────
class Solution(Bot):
    def __init__(self):
        super().__init__()
        self.learned_map  = []                    
        self._global_meas = np.zeros((0, 2))
        self._assoc       = np.array([], dtype=int)
        
        # --- Landmark Lifecycle States ---
        self.map_states = []  
        self.map_counts = []  

    def mapping(self, measurements):
        if len(measurements) == 0:
            return
            
        gm = local_to_global(measurements, self.pos, self.heading)
        
        CONFIRM_THRESH = 3
        MERGE_DIST     = 2.0
        KALMAN_GAIN    = 0.25  

        for p in gm:
            if not self.map_states:
                self.map_states.append(p.copy())
                self.map_counts.append(1)
                continue

            dists = [np.linalg.norm(p - q) for q in self.map_states]
            min_idx = np.argmin(dists)

            if dists[min_idx] < MERGE_DIST:
                self.map_states[min_idx] += KALMAN_GAIN * (p - self.map_states[min_idx])
                self.map_counts[min_idx] += 1
            else:
                self.map_states.append(p.copy())
                self.map_counts.append(1)

        self.learned_map = [
            state for state, count in zip(self.map_states, self.map_counts)
            if count >= CONFIRM_THRESH
        ]

# ── Problem 3 – Mapping ───────────────────────────────────────────────────────
def make_problem3():
    sol = Solution()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Problem 3 – Mapping  (EMA Filter + Lifecycle)",
                 fontsize=13, fontweight="bold")

    def update(frame):
        ax.clear()
        steer = pure_pursuit(sol.pos, sol.heading, CENTERLINE)
        meas  = get_measurements(sol.pos, sol.heading)
        sol.pos, sol.heading = step_kinematic(sol.pos, sol.heading, SPEED, steer)
        sol.mapping(meas)

        draw_track(ax, alpha_b=0.15, alpha_y=0.15)

        if sol.learned_map:
            lm = np.array(sol.learned_map)
            ax.scatter(lm[:, 0], lm[:, 1],
                       c="limegreen", marker="x", s=90, linewidths=2.0,
                       zorder=5, label=f"Mapped cones ({len(lm)})")

        draw_car(ax, sol.pos, sol.heading)
        setup_ax(ax,
            f"Frame {frame+1}/{N_FRAMES}  –  "
            f"map size: {len(sol.learned_map)} / {len(MAP_CONES)} cones")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, repeat=True)
    return fig, ani


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Driverless Car Hackathon – SLAM Visualisation ===")
    fig3, ani3 = make_problem3()
    plt.show()
