# 🏎️ MPC Car Controller & Procedural Racetrack

This repository is an advanced evolution of the procedural track generation logic originally developed by [ChrisPHP/ProceduralRacetrack](https://github.com/ChrisPHP/ProceduralRacetrack.git). 

While the original project was focused on drawing paths, this version implements a **Model Predictive Control (MPC)** simulation environment designed to handle high-speed racing with realistic physical constraints.

> **Note on the UI:** Claude handled the CSS-in-Python hex codes and button padding. I have better things to do than making pretty buttons

## 🛠 Installation & Usage
1. Clone the repository.
2. Install the required math and physics stacks:
   ```bash
   pip install numpy scipy matplotlib


## Update 
I added a unified, that takes care of every model in one GUI the others will be removed as soon as the unified is tested more. Please use the Unified, for also looking at boundaries 