# MPC Car Controller & Procedural Racetrack Extended

This repository is an advanced extension of the procedural track generation logic originally developed by [ChrisPHP/ProceduralRacetrack](https://github.com/ChrisPHP/ProceduralRacetrack.git). 

While the original project focused on track generation, this version introduces a **Model Predictive Control (MPC)** simulation environment with a sophisticated, real-time interactive GUI.

> **Note on the UI:** The GUI layout and widget styling were handled by Claude. I have better things to do than manually positioning buttons and writing hex codes, so I let the AI handle the boilerplate while I focused on the control logic and simulation physics.

## 🚀 Key Enhancements

### 1. Real-Time MPC Engine
I’ve integrated a full Model Predictive Control solver using the `SLSQP` method. 
* **Dynamic Horizon:** Adjust the prediction horizon ($N$) on the fly to see how the car's "look-ahead" behavior changes.
* **Bicycle Model Dynamics:** Realistic physics including wheelbase ($L$) effects and non-linear constraints.
* **Multi-Objective Tuning:** The controller balances cross-track error, heading alignment, velocity tracking, and control effort (smoothness) penalties.

### 2. Interactive Control Dashboard
A custom dark-themed GUI that allows for "Scientific Tuning" of the vehicle:
* **Precision Number Inputs:** Replaced generic sliders with `Spinbox` controls for exact parameter entry (weights like $0.001$ actually matter).
* **Live Telemetry:** Real-time readout of velocity, steering angle ($\\delta$), acceleration ($a$), and optimizer cost.
* **On-the-fly Physics:** Change vehicle wheelbase, noise σ, or max speed limits while the car is mid-lap.

### 3. Smart Track & Path Logic
* **Auto-Lap Reset:** The driven path (breadcrumb trail) automatically clears itself every time the car crosses the start/finish line, keeping the canvas clean for long-term testing.
* **Speed Profiling:** The system generates an optimal speed profile based on track curvature—the car intelligently slows down for hairpins and guns it on the straights.
* **Track Persistence:** Save your procedurally generated tracks as `.csv` or `.npy` and reload them later.

## 🛠 Installation & Usage
1. Clone this repository.
2. Install the required math and physics stacks:
   ```bash
   pip install numpy scipy matplotlib

3. Run mpc_car_gui.py
