# 🦾 Robot Safety Simulation

A Python-based simulation of a **fenceless robot safety system**. It generates synthetic 3D sensor data, transforms detections from the **sensor frame** to the **robot base frame** using **homogeneous transformation matrices**, computes the **closest approach distance**, and classifies each frame into safety states: **NORMAL**, **SLOW**, or **STOP**. Includes Matplotlib visualizations (distance-over-time + 3D scatter).

---

## ✨ Features
- **Synthetic point clouds** that simulate a person approaching the robot.
- **3D transforms**: rotation matrices (Rx/Ry/Rz) + 4×4 homogeneous transforms.
- **Safety thresholds**: configurable `SLOW_RADIUS` and `STOP_RADIUS`.
- **Visualizations**:
  - Line plot: minimum distance vs. frame with threshold overlays.
  - 3D scatter in the base frame (marks the robot base).
 
