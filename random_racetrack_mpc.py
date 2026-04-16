import numpy as np
from scipy.interpolate import interp1d
from createtrack import CreateTrack
import matplotlib.pyplot as plt

# Assuming 'track_raw' is the output from your CreateTrack logic:
# track_raw = creator.create_racetrack(track_3d=False)

def resample_track_for_mpc(track_raw, step_distance=1.0):
    """
    Takes raw Bezier curves and resamples them so each point 
    is exactly 'step_distance' apart.
    """
    track_raw = np.array(track_raw)
    # 1. Ensure the track is a closed loop (connect last point to first)
    if not np.allclose(track_raw[0], track_raw[-1]):
        track_raw = np.vstack([track_raw, track_raw[0]])

    # 2. Calculate cumulative distance (Arc Length) along the Bezier path
    # This turns the "index" of the points into "meters"
    dw = np.diff(track_raw, axis=0)
    step_lengths = np.sqrt(np.sum(dw**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(step_lengths), 0, 0)
    total_length = cumulative_dist[-1]

    # 3. Create a linear range of distances [0, 1, 2, ... Total]
    # This ensures the MPC has a consistent reference speed/distance
    distance_interp = np.arange(0, total_length, step_distance)

    # 4. Interpolate X and Y based on the new distance array
    f_x = interp1d(cumulative_dist, track_raw[:, 0], kind='linear')
    f_y = interp1d(cumulative_dist, track_raw[:, 1], kind='linear')

    # 5. Generate final MPC-ready track
    track_mpc = np.vstack([f_x(distance_interp), f_y(distance_interp)]).T
    
    return track_mpc

def generate_track_for_mpc(num_points=12, x_bounds=[0, 200], y_bounds=[0, 200], corner_cells=20, step_distance=1.0):
    creator = CreateTrack(num_points=num_points, x_bounds=x_bounds, y_bounds=y_bounds, corner_cells=corner_cells)
    raw_points = creator.create_racetrack(track_3d=False)
    final_track = resample_track_for_mpc(raw_points, step_distance=step_distance)
    return final_track

