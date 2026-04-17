# 🏎️ MPC Car Controller & Procedural Racetrack

This repository is an advanced evolution of the procedural track generation logic originally developed by [ChrisPHP/ProceduralRacetrack](https://github.com/ChrisPHP/ProceduralRacetrack.git). 

While the original project was focused on drawing paths, this version implements a high-performance **Model Predictive Control (MPC)** simulation environment designed to handle high-speed racing with realistic physical constraints.

> **Note on the UI:** Claude handled the CSS-in-Python hex codes and button padding. I have better things to do than making pretty buttons

## 🚀 Key Enhancements

### 1. Dynamic 4-Wheel Physics Model
Moved beyond simple geometry to a physics-informed model that respects tire limits and numerical stability:
* **Slip Angle Awareness:** The controller calculates front and rear slip angles ($\alpha_f, \alpha_r$) using `arctan2` for stable handling even during high-lateral-G maneuvers.
* **Linear Tire Forces:** Uses a lateral stiffness coefficient ($C_{\alpha} \approx 80,000$) to simulate tire grip. Forces are saturated within the optimizer to prevent "slingshot" physics.
* **Numerical Safety (Anti-Warp Governor):** Implemented hard-state clipping inside the prediction loop. The car can no longer "hallucinate" impossible speeds to minimize cost; it is physically bound to the user-defined limits.

### 2. Intelligent Look-Ahead Logic
The MPC has been upgraded with "Physical Foresight" to prevent the common issue of over-acceleration out of slow corners:
* **Distance-Based Indexing:** Instead of looking $N$ indices ahead, the controller projects reference points based on **physical distance** (meters).
* **The 5-Meter Floor:** Implemented a minimum look-ahead distance. Even at $0$ m/s, the car "reaches" 5 meters down the track, ensuring it always has a stable heading target.
* **Velocity Sync:** The velocity error ($verr$) is synced to the projected distance index, preventing the car from trying to match "straightaway speeds" while still in the apex of a turn.

### 3. Advanced Cost Function & Stability
* **The Friction Circle Penalty:** Added a **Longitudinal-Lateral Coupling Penalty** ($a^2 \cdot \delta^2$). The car intelligently rolls off the throttle when steering hard to maximize available cornering grip.
* **Dynamic Weight Scaling:** Steering stability weights ($R_s$) scale quadratically with velocity. As speed increases, the steering "stiffens" to prevent high-speed oscillations.
* **Control Smoothness:** Heavy penalties on change-of-acceleration ($Rd_a$) and change-of-steering ($Rd_s$) ensure professional-grade, smooth inputs.

### 4. Interactive "Scientific" GUI
* **Dual Horizon Visualization:** The UI renders two distinct data sets for real-time debugging:
    1.  **The Plan (Blue Line):** The car’s predicted physical trajectory over the horizon.
    2.  **The Targets (Red Dots):** The actual track waypoints the car is attempting to chase, honoring the minimum look-ahead logic.
* **Live Telemetry:** Real-time readout of $v, a, \delta,$ and total optimizer cost.
* **Spinbox Precision:** Replaced generic sliders with precision `Spinbox` controls for fine-tuning weights where $0.001$ makes the difference between a lap record and a crash.

## 🛠 Installation & Usage
1. Clone the repository.
2. Install the required math and physics stacks:
   ```bash
   pip install numpy scipy matplotlib


## Update 
I added a unified, that takes care of every model in one GUI the others will be removed as soon as the unified is tested more. Please use the Unified, for also looking at boundaries 