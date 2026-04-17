import numpy as np
from scipy.interpolate import interp1d
from createtrack import CreateTrack
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev

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
            # Random step between 0.5m and 5m
            step = np.random.uniform(0.5, 5.0)
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



def generate_track_boundaries(midpoints, width=2.0, resolution=100):
    """
    Generate left and right track boundaries from midpoints (centerline).

    Args:
        midpoints: Array of (x,y) points defining the track centerline
        width: Track width in meters (distance between boundaries)
        resolution: Number of points to use for smooth boundaries

    Returns:
        left_boundary, right_boundary: Two arrays of (x,y) points
    """
    # Close the loop if not already closed
    if not np.allclose(midpoints[0], midpoints[-1]):
        midpoints = np.vstack([midpoints, midpoints[0]])

    # Fit a spline to the midpoints
    tck, u = splprep(midpoints.T, s=0, per=True)

    # Generate smooth centerline points
    u_smooth = np.linspace(0, 1, resolution)
    x, y = splev(u_smooth, tck)
    centerline = np.column_stack((x, y))

    # Calculate tangents and normals
    dx = np.gradient(x)
    dy = np.gradient(y)
    length = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero
    length[length == 0] = 1e-6

    # Normalize tangent vectors
    nx = -dy / length  # Left normal (pointing inward)
    ny = dx / length   # Left normal (pointing inward)

    # Generate boundaries
    half_width = width / 2
    left_boundary = centerline + half_width * np.column_stack((nx, ny))
    right_boundary = centerline - half_width * np.column_stack((nx, ny))

    return left_boundary, right_boundary

def resample_boundaries(left, right, step_distance=1.0):
    """
    Resample both boundaries to have consistent point spacing.

    Args:
        left: Left boundary points
        right: Right boundary points
        step_distance: Target distance between points

    Returns:
        Resampled left and right boundaries
    """
    # Combine both boundaries for resampling
    combined = np.vstack([left, left[0], right, right[0]])

    # Calculate cumulative distance
    dw = np.diff(combined, axis=0)
    step_lengths = np.sqrt(np.sum(dw**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(step_lengths), 0, 0)
    total_length = cumulative_dist[-1]

    # Create distance array
    distance_interp = np.arange(0, total_length, step_distance)

    # Interpolate
    f = interp1d(cumulative_dist, combined, axis=0, kind='linear')
    resampled = f(distance_interp)

    # Split back into left and right boundaries
    split_idx = len(resampled) // 2
    resampled_left = resampled[:split_idx]
    resampled_right = resampled[split_idx:-1]  # Remove duplicate closing point

    return resampled_left, resampled_right


def generate_track_for_mpc(num_points=12, x_bounds=[0, 200], y_bounds=[0, 200], corner_cells=20, step_distance=1.0):
    creator = CreateTrack(num_points=num_points, x_bounds=x_bounds, y_bounds=y_bounds, corner_cells=corner_cells)
    raw_points = creator.create_racetrack(track_3d=False)
    final_track = resample_track_for_mpc(raw_points, step_distance=step_distance)
    return final_track

def generate_random_track_for_mpc_with_boundaries(num_points=12, x_bounds=[0, 200], y_bounds=[0, 200], corner_cells=20, step_distance=1.0, track_width=3.0):
    midpoints = generate_track_for_mpc(num_points=num_points, x_bounds=x_bounds, y_bounds=y_bounds, corner_cells=corner_cells, step_distance=step_distance)
    left_boundary, right_boundary = generate_track_boundaries(midpoints, width=track_width)
    left_resampled, right_resampled = resample_boundaries(left_boundary, right_boundary, step_distance)
    return midpoints, left_resampled, right_resampled

