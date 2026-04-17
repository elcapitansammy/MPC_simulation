import numpy as np 

class CarDynamics4Wheel:
    def __init__(self, x=0.0, y=0.0, v=0.0, theta=0.0, L=2.5, Lf=1.2, Lr=1.3, m=1500, Iz=3000, mu=0.8, noise_a=0.1, noise_steer_deg=1.0, max_accel=9.0, max_decel=15.0, max_steer=0.7, max_speed=40.0):
        # Vehicle parameters
        self.L = L          # Wheelbase [m]
        self.Lf = Lf        # Distance from CG to front axle [m]
        self.Lr = Lr        # Distance from CG to rear axle [m]
        self.m = m          # Mass [kg]
        self.Iz = Iz        # Yaw inertia [kg*m^2]
        self.mu = mu        # Friction coefficient
        self.noise_a = noise_a
        self.noise_steer_deg = noise_steer_deg
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.max_steer = max_steer
        self.max_speed = max_speed

        # State [x, y, yaw, v, w]
        self.x = x
        self.y = y
        self.theta = theta   # yaw angle
        self.v = v          # velocity [m/s]
        self.w = 0.0        # yaw rate [rad/s]

        # Tire parameters (linear model)
        self.Ca_f = 80000   # Front cornering stiffness [N/rad]
        self.Ca_r = 80000   # Rear cornering stiffness [N/rad]
        self.mu = mu        # Friction coefficient
        self.g = 9.81       # Gravity [m/s^2]

    def update(self, a, delta, dt):
        # 1. Apply noise if needed
        na = np.random.normal(0, self.noise_a)
        ns = np.random.normal(0, np.radians(self.noise_steer_deg))
        
        # These represent the 'real' commands reaching the wheels
        a_noisy = a + na
        delta_noisy = delta + ns

        # 2. Match the MPC Clipping (CRITICAL)
        # This prevents the simulation from going Mach 2 even if the MPC fails
        a_final = np.clip(a_noisy, -self.max_decel, self.max_accel)  # Use your accel limits here
        delta_final = np.clip(delta_noisy, -self.max_steer, self.max_steer)

        # 3. Calculate Physics (Exactly like the cost function)
        # v_new is calculated FIRST to be used in slip and position
        self.v = np.clip(self.v + a_final * dt, 0.1, self.max_speed + self.max_speed/10) 

        # Current slip angles
        beta = np.arctan2(self.Lr * self.w, self.v)
        alpha_f = delta_final - beta
        alpha_r = -np.arctan2(self.Lr * self.w, self.v)

        # Linear tire forces (Matching the 80k constant in your MPC)
        Fy_f = -80000 * alpha_f
        Fy_r = -80000 * alpha_r

        # 4. Integrate State
        # Note: We use the v that was just updated (self.v)
        self.w += (self.Lf * Fy_f * np.cos(delta_final) - self.Lr * Fy_r) / 3000 * dt
        self.theta += self.w * dt
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt

    def get_state(self):
        return self.x, self.y, self.theta, self.v, self.w
    
# # Front tires (left + right, assume equal load)
        # Fy_f = 2 * pacejka(alpha_f, Fz=4000, B=10.0, C=1.5, mu=0.8)

        # # Rear tires
        # Fy_r = 2 * pacejka(alpha_r, Fz=4000, B=10.0, C=1.5, mu=0.8)    
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def test_car_model():
    # Initialize car with some parameters
    car = CarDynamics4Wheel(L=2.5, Lf=1.2, Lr=1.3, m=1500,
                           noise_a=0, noise_steer_deg=0)

    # Simulation parameters
    dt = 0.02
    total_time = 10.0
    steps = int(total_time / dt)

    # Inputs (acceleration and steering angle)
    # We'll do a step steer input (5 degrees at t=1s)
    def get_inputs(t):
        if t < 3.0:
            return 1, np.radians(0.0)  # Accelerate + 5° left turn

        # Phase 2: Decelerate while turning right (3-6s)
        elif t < 3.5:
            return -0.1, np.radians(5)  # Decelerate + 5° right turn

        # Phase 3: Come to complete stop (6s+)
        else:
            return 0.0, np.radians(0.0)  # Gentle brake to stop

    # Store history for plotting
    history = []
    time_history = []

    # Run simulation
    for i in range(steps):
        t = i * dt
        a, delta = get_inputs(t)
        car.update(a, delta, dt)
        history.append(car.get_state())
        time_history.append(t)

    # Convert history to numpy array
    history = np.array(history)
    x, y, theta, v, w = history.T

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot trajectory
    plt.subplot(2, 2, 1)
    plt.plot(x, y, 'b-', label='Trajectory')
    plt.scatter(x[0], y[0], c='g', label='Start')
    plt.scatter(x[-1], y[-1], c='r', label='End')
    plt.xlabel('X position [m]')
    plt.ylabel('Y position [m]')
    plt.title('Vehicle Trajectory')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    # Plot velocity
    plt.subplot(2, 2, 2)
    plt.plot(time_history, v, 'r-')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity Profile')
    plt.grid(True)

    # Plot yaw rate
    plt.subplot(2, 2, 3)
    plt.plot(time_history, np.degrees(w), 'g-')
    plt.xlabel('Time [s]')
    plt.ylabel('Yaw rate [deg/s]')
    plt.title('Yaw Rate')
    plt.grid(True)

    # Plot steering angle
    steering_angles = [np.degrees(get_inputs(t)[1]) for t in time_history]
    plt.subplot(2, 2, 4)
    plt.plot(time_history, steering_angles, 'm-')
    plt.xlabel('Time [s]')
    plt.ylabel('Steering angle [deg]')
    plt.title('Steering Input')
    plt.grid(True)

    plt.tight_layout()

    # Show some vehicle poses
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'b-', alpha=0.3)

    # Draw vehicle at several points
    for i in [0, 50, 100, 200, 300, -1]:
        draw_car(x[i], y[i], theta[i], car.L, car.Lf, car.Lr, color='r', alpha=0.5)

    plt.title('Vehicle Positions During Maneuver')
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    
def pacejka(alpha, Fz=4000, B=12.0, C=1.5, D=1.0, E=-1.0, mu=0.8):
    Fy_max = mu * Fz  # e.g., 0.8 * 4000 = 3200 N per tire
    Fy = D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))))
    return Fy * Fy_max  # Scale by Fy_max to get realistic forces (e.g., 3200 N at peak)

def draw_car(x, y, theta, L, Lf, Lr, color='r', alpha=1.0):
    """Helper function to draw a car as a rectangle with wheels"""
    car_width = L * 0.6  # Make car width 60% of wheelbase

    # Create rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Define car corners in local coordinates
    corners_local = np.array([
        [Lf, car_width/2],   # Front right
        [Lf, -car_width/2],  # Front left
        [-Lr, -car_width/2], # Rear left
        [-Lr, car_width/2]   # Rear right
    ])

    # Rotate and translate corners
    corners = np.dot(corners_local, R.T) + np.array([x, y])

    # Draw car body
    plt.fill(corners[:, 0], corners[:, 1], color=color, alpha=alpha)

    # Draw front wheels
    for side in [-1, 1]:
        wheel_pos = np.array([Lf, side * car_width/2 * 0.8])
        wheel_pos = np.dot(wheel_pos, R.T) + np.array([x, y])
        plt.plot(wheel_pos[0], wheel_pos[1], 'ko', markersize=3)

    # Draw rear wheels
    for side in [-1, 1]:
        wheel_pos = np.array([-Lr, side * car_width/2 * 0.8])
        wheel_pos = np.dot(wheel_pos, R.T) + np.array([x, y])
        plt.plot(wheel_pos[0], wheel_pos[1], 'ko', markersize=3)