import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots

# Rotation Matrices
def Rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

# Homogenous Transformation Matrix

def make_T(R, t3):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t3
    return T

# Set up - B is base frame and S is sensor frame   

R_BS = Rz(np.deg2rad(12.0)) @ Ry(np.deg2rad(-6.0)) # rotate sensor frame relative to base frame
t_BS = np.array([0.35, 0.10, 0.95]) # translate sensor frame relative to base frame
T_BS = make_T(R_BS, t_BS) # Make the transformation matrix

# Safety Thresholds
SLOW_RADIUS = 2.2
STOP_RADIUS = 1.8

# Generate sensor detections

def synthetic_sensor_points(frame_idx, n=60, seed=42):
    rng = np.random.default_rng(seed + frame_idx)
    mean = np.array([1.2 - 0.02*frame_idx, 0.0, 1.1])
    cov = np.diag([0.15, 0.10, 0.10])**2
    pts = rng.multivariate_normal(mean, cov, size=n)  # (n,3) in Sensor frame
    return pts

def to_homogeneous(pts):
    return np.c_[pts, np.ones(len(pts))].T

# Main Loop
def eval_frame(frame_idx):
    pts_S = synthetic_sensor_points(frame_idx)
    P = to_homogeneous(pts_S)            # (4,n)
    P_B = T_BS @ P                        # Sensor -> Base frame
    xyz_B = P_B[:3, :].T                  # (n,3) drop homogeneous row

    # Distance to base origin (0,0,0)
    dists = np.linalg.norm(xyz_B, axis=1)
    dmin = float(dists.min())

    # Safety state
    if dmin < STOP_RADIUS:
        state = "STOP"
    elif dmin < SLOW_RADIUS:
        state = "SLOW"
    else:
        state = "NORMAL"
    return state, dmin, xyz_B

if __name__ == "__main__":
    print("Starting safety robot simulation")
    for k in range(200):  # 200 frames (~20 seconds)
        state, dmin, _ = eval_frame(k)
        print(f"frame {k:03d} | min_dist={dmin:0.3f} m | state={state}")
        time.sleep(0.1)

NUM_FRAMES = 200
results = [eval_frame(k) for k in range(NUM_FRAMES)]

# 1) Extract values for plotting
frames = list(range(NUM_FRAMES))          # 0..199
dmins  = [r[1] for r in results]          # min distance
states = [r[0] for r in results]          # "NORMAL"/"SLOW"/"STOP"

# ---------- Plot 1: Distance vs. Frame ----------
plt.figure(figsize=(8,4.5))
plt.plot(frames, dmins, label="Min distance (m)")
plt.axhline(SLOW_RADIUS, linestyle="--", color="orange", label=f"SLOW = {SLOW_RADIUS} m")
plt.axhline(STOP_RADIUS, linestyle=":", color="red", label=f"STOP = {STOP_RADIUS} m")
plt.xlabel("Frame")
plt.ylabel("Closest detected distance (m)")
plt.title("Safety distance over time")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- Plot 2: 3D Scatter at the most critical frame ----------
idx_min = int(np.argmin(dmins))             # frame with smallest distance
state_best, dmin_best, xyz_B = results[idx_min]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz_B[:,0], xyz_B[:,1], xyz_B[:,2], s=15, alpha=0.6)
ax.scatter([0],[0],[0], color="red", marker="x", s=100, label="Robot base")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(f"Frame {idx_min}: min={dmin_best:.2f} m | state={state_best}")
ax.legend()
plt.tight_layout()
plt.show()


