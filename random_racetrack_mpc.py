import numpy as np
from scipy.interpolate import interp1d
from createtrack import CreateTrack
import matplotlib.pyplot as plt

# Assuming 'track_raw' is the output from your CreateTrack logic:
# track_raw = creator.create_racetrack(track_3d=False)

def resample_track_for_mpc(track_raw, step_distance=1.0):
    """
    Takes raw track points and resamples them so each point
    is exactly 'step_distance' apart. If step_distance is -1,
    uses random distances between 0.5m and 5m.

    Args:
        track_raw: Array of (x,y) points defining the track
        step_distance: Distance between points (meters). If -1, uses random distances.

    Returns:
        Resampled track with consistent or random spacing
    """
    track_raw = np.array(track_raw)

    # 1. Ensure the track is a closed loop
    if not np.allclose(track_raw[0], track_raw[-1]):
        track_raw = np.vstack([track_raw, track_raw[0]])

    # 2. Calculate cumulative distance along the path
    dw = np.diff(track_raw, axis=0)
    step_lengths = np.sqrt(np.sum(dw**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(step_lengths), 0, 0)
    total_length = cumulative_dist[-1]

    # 3. Create distance array - either fixed or random steps
    if step_distance == -1:
        # Generate random distances between 0.5m and 5m
        distance_interp = [0.0]  # Start at 0
        current_dist = 0.0

        while current_dist < total_length:
            # Random step between 0.5m and 10m
            step = np.random.uniform(0.5, 10.0)
            current_dist += step
            if current_dist < total_length:
                distance_interp.append(current_dist)

        distance_interp = np.array(distance_interp)
    else:
        # Fixed step distance
        distance_interp = np.arange(0, total_length, step_distance)

    # 4. Interpolate X and Y based on the distance array
    f_x = interp1d(cumulative_dist, track_raw[:, 0], kind='linear')
    f_y = interp1d(cumulative_dist, track_raw[:, 1], kind='linear')

    # 5. Generate final track
    track_mpc = np.vstack([f_x(distance_interp), f_y(distance_interp)]).T

    return track_mpc

def generate_track_for_mpc(num_points=12, x_bounds=[0, 200], y_bounds=[0, 200], corner_cells=20, step_distance=1.0):
    creator = CreateTrack(num_points=num_points, x_bounds=x_bounds, y_bounds=y_bounds, corner_cells=corner_cells)
    raw_points = creator.create_racetrack(track_3d=False)
    final_track = resample_track_for_mpc(raw_points, step_distance=step_distance)
    return final_track

