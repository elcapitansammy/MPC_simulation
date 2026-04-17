"""
MPC Car Controller — Unified Version
─────────────────────────────────────
Combines Kinematic & Dynamic models, boundary-aware MPC, 
advanced visualization, and full track/car parameter control.

Controls: 
- Start/Pause/Reset
- Model Selection (Kinematic vs Dynamic)
- Track Generation (Width, Size, Complexity)
- Boundary-Aware MPC Toggle
- Live Parameter Tuning
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import patches
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.spatial import KDTree

# Try importing the external track generator
try:
    from random_racetrack_mpc import generate_track_for_mpc, generate_track_boundaries, resample_track_for_mpc
    _HAS_EXT_TRACK = True
except ImportError:
    _HAS_EXT_TRACK = False

# ═══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class CarDynamics4Wheel:
    def __init__(self, x=0.0, y=0.0, v=0.0, theta=0.0, L=2.5, Lf=1.2, Lr=1.3, m=1500, Iz=3000, mu=0.8, noise_a=0.1, noise_steer_deg=1.0, max_accel=9.0, max_decel=15.0, max_steer=0.7, max_speed=40.0, **kwargs):
        self.L, self.Lf, self.Lr = L, Lf, Lr
        self.m, self.Iz, self.mu = m, Iz, mu
        self.noise_a = noise_a
        self.noise_steer_deg = noise_steer_deg
        self.max_accel, self.max_decel = max_accel, max_decel
        self.max_steer, self.max_speed = max_steer, max_speed
        self.x, self.y, self.theta, self.v = x, y, theta, v
        self.w = 0.0
        self.Ca_f = kwargs.get('Ca_f', 80000.0)
        self.Ca_r = kwargs.get('Ca_r', 80000.0)

    def update(self, a, delta, dt):
        na = np.random.normal(0, self.noise_a)
        ns = np.random.normal(0, np.radians(self.noise_steer_deg))
        a_eff = np.clip(a + na, -self.max_decel, self.max_accel)
        delta_eff = np.clip(delta + ns, -self.max_steer, self.max_steer)
        self.v = np.clip(self.v + a_eff * dt, 0.1, self.max_speed * 1.1)
        beta = np.arctan2(self.Lr * self.w, self.v)
        af, ar = delta_eff - beta, -np.arctan2(self.Lr * self.w, self.v)
        Ff, Fr = -self.Ca_f * af, -self.Ca_r * ar
        self.w += (self.Lf * Ff * np.cos(delta_eff) - self.Lr * Fr) / self.Iz * dt
        self.theta += self.w * dt
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt

    def get_state(self):
        return self.x, self.y, self.theta, self.v, self.w

class KinematicBicycleModel:
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0, L=1.0, noise_a=0.2, noise_steer_deg=1.0, **kwargs):
        self.x, self.y, self.theta, self.v = x, y, theta, v
        self.L = L
        self.noise_a = noise_a
        self.noise_steer_deg = noise_steer_deg
        self.max_accel = kwargs.get('max_accel', 9.0)
        self.max_decel = kwargs.get('max_decel', 15.0)
        self.max_steer = kwargs.get('max_steer', 0.7)
        self.max_speed = kwargs.get('max_speed', 40.0)

    def update(self, a, delta, dt):
        na = np.random.normal(0, self.noise_a)
        ns = np.random.normal(0, np.radians(self.noise_steer_deg))
        
        a_eff = np.clip(a + na, -self.max_decel, self.max_accel)
        d_eff = np.clip(delta + ns, -self.max_steer, self.max_steer)
        
        self.v = np.clip(self.v + a_eff * dt, 0.1, self.max_speed * 1.1)
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += (self.v / self.L) * np.tan(d_eff) * dt

    def get_state(self):
        return self.x, self.y, self.theta, self.v, 0.0 # w=0 for kinematic

# ═══════════════════════════════════════════════════════════════════════════════
#  TRACK UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def _resample(points, step_distance):
    diffs = np.diff(points, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.insert(np.cumsum(seg_len), 0, 0)
    total = cum[-1]
    ds = np.arange(0, total, step_distance)
    fx = interp1d(cum, points[:, 0], kind="linear")
    fy = interp1d(cum, points[:, 1], kind="linear")
    return np.vstack([fx(ds), fy(ds)]).T

def _builtin_track(step_distance=3.0, n_control=12, seed=None, x_size=200, y_size=200):
    rng = np.random.default_rng(seed if (seed is not None and seed >= 0) else None)
    angles = np.linspace(0, 2 * np.pi, n_control, endpoint=False)
    radii = rng.uniform(min(x_size, y_size)*0.2, min(x_size, y_size)*0.4, n_control)
    jitter = rng.uniform(-0.25, 0.25, n_control)
    cx = radii * np.cos(angles + jitter) + x_size/2
    cy = radii * np.sin(angles + jitter) + y_size/2
    cx = np.append(cx, cx[0])
    cy = np.append(cy, cy[0])
    t = np.linspace(0, 1, len(cx))
    t_fine = np.linspace(0, 1, 3000)
    fine = np.vstack([interp1d(t, cx, kind="cubic")(t_fine), interp1d(t, cy, kind="cubic")(t_fine)]).T
    return _resample(fine, step_distance)

def generate_boundaries(midpoints, width=5.0):
    pts = np.vstack([midpoints, midpoints[0]])
    dx = np.gradient(pts[:, 0]); dy = np.gradient(pts[:, 1])
    length = np.sqrt(dx**2 + dy**2); length[length == 0] = 1e-6
    nx = -dy / length; ny = dx / length
    half_width = width / 2
    left = pts + half_width * np.column_stack((nx, ny))
    right = pts - half_width * np.column_stack((nx, ny))
    return left[:-1], right[:-1]

def get_speed_profile(track, max_speed, min_speed=3.0, curvature_gain=5.0, lookahead=10):
    curvatures = np.zeros(len(track))
    if len(track) < 3: return np.full(len(track), min_speed)
    for i in range(len(track)):
        p0, p1, p2 = track[i-1], track[i], track[(i+1) % len(track)]
        v1, v2 = p1 - p0, p2 - p1
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: continue
        cos_t = np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)
        curvatures[i] = 2 * np.sin(np.arccos(cos_t)/2) / (np.linalg.norm(p2-p0)/2 + 1e-6)
    
    extended_curv = np.copy(curvatures)
    for i in range(len(track)):
        indices = [(i + j) % len(track) for j in range(lookahead + 1)]
        extended_curv[i] = max(curvatures[indices])
    
    v = max_speed / (1 + curvature_gain * extended_curv)
    return np.clip(v, min_speed, max_speed)

# ═══════════════════════════════════════════════════════════════════════════════
#  MPC COST
# ═══════════════════════════════════════════════════════════════════════════════

def mpc_cost(u_flat, state, model_type, params, track, left_track, right_track, use_bounds, N, dt, last_u, vprof, weights):
    # weights: Q_dist, Q_psi, Q_v, R_a, R_s, Rd_a, Rd_s, Q_bound
    Q_dist, Q_psi, Q_v, R_a, R_s, Rd_a, Rd_s, Q_bound = weights
    controls = u_flat.reshape(-1, 2)
    x, y, th, v, w = state
    
    # Params
    L = params.get('L', 1.0)
    Lf = params.get('Lf', 1.2)
    Lr = params.get('Lr', 1.3)
    Iz = params.get('Iz', 3000.0)
    Ca_f = params.get('Ca_f', 80000.0)
    Ca_r = params.get('Ca_r', 80000.0)
    max_a = params.get('max_accel', 9.0)
    max_d = params.get('max_decel', 15.0)
    max_s = params.get('max_steer', 0.7)
    max_v = params.get('max_speed', 40.0)
    min_la = params.get('min_lookahead', 3.0)

    dists = np.sum((track - np.array([x, y])) ** 2, axis=1)
    idx = int(np.argmin(dists))
    
    total = 0.0
    pa, pd = last_u
    beta = np.arctan2(Lr * w, v) if v > 0.1 else 0.0
    
    p1, p2 = track[idx], track[(idx + 1) % len(track)]
    dist_per_idx = max(np.linalg.norm(p2 - p1), 0.1)

    for i in range(N):
        a, delta = controls[i]
        a = np.clip(a, -max_d, max_a)
        delta = np.clip(delta, -max_s, max_s)

        v_next = np.clip(v + a * dt, 0.1, max_v * 1.1)
        
        if model_type == "Dynamic":
            af, ar = delta - beta, -np.arctan2(Lr * w, v_next)
            Ff, Fr = -Ca_f * af, -Ca_r * ar
            w_next = w + (Lf * Ff * np.cos(delta) - Lr * Fr) / Iz * dt
            th_next = th + w * dt
            x_next = x + v_next * np.cos(th) * dt
            y_next = y + v_next * np.sin(th) * dt
            beta = np.arctan2(Lr * w_next, v_next) if v_next > 0.1 else 0.0
        else: # Kinematic
            x_next = x + v_next * np.cos(th) * dt
            y_next = y + v_next * np.sin(th) * dt
            th_next = th + (v_next / L) * np.tan(delta) * dt
            w_next = 0.0

        # Dynamic effort weights
        Rs_dyn = R_s * max(1.0, (v_next/30.0)**2)
        Rds_dyn = Rd_s * max(1.0, (v_next/30.0)**2)

        # Reference
        la_dist = max(min_la, v_next * (dt * (i + 1)))
        ri = (idx + int(round(la_dist / dist_per_idx))) % len(track)
        ni = (ri + 1) % len(track)
        
        ref = track[ri]
        target_th = np.arctan2(track[ni, 1] - ref[1], track[ni, 0] - ref[0])

        dsq = (x_next - ref[0])**2 + (y_next - ref[1])**2
        herr = (th_next - target_th + np.pi) % (2 * np.pi) - np.pi
        verr = (v_next - vprof[ri])**2

        total += Q_dist * dsq + Q_psi * herr**2 + Q_v * verr
        total += R_a * a**2 + Rs_dyn * delta**2
        total += Rd_a * (a - pa)**2 + Rds_dyn * (delta - pd)**2

        if use_bounds and left_track is not None:
            ref_l, ref_r = left_track[ri], right_track[ri]
            v_rl = ref_l - ref_r; w_sq = np.sum(v_rl**2)
            if w_sq > 0.1:
                t_proj = np.dot(np.array([x_next, y_next]) - ref_r, v_rl) / w_sq
                total += Q_bound * (max(0, -t_proj)**2 + max(0, t_proj - 1)**2)

        x, y, th, v, w, pa, pd = x_next, y_next, th_next, v_next, w_next, a, delta

    return total

# ═══════════════════════════════════════════════════════════════════════════════
#  GUI & APP
# ═══════════════════════════════════════════════════════════════════════════════

C = {
    "bg": "#0d0d14", "panel": "#13131f", "card": "#1a1a2e", "border": "#252542",
    "accent1": "#00e5ff", "accent2": "#ff4d6d", "accent3": "#b5ff4d",
    "text": "#c8c8e0", "muted": "#5a5a88", "white": "#f0f0ff",
    "slider": "#1e1e38", "trough": "#2a2a48", "tab_sel": "#20203a", "tab_off": "#16162a",
}

class MPCCarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MPC Unified Car Controller")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1280x800")

        self.running = False
        self.frame_num = 0
        self.hist_x, self.hist_y = [], []
        self.track = self.left_track = self.right_track = None
        self.car = None
        self.u_guess = np.zeros(10)
        self.last_u = np.array([0.0, 0.0])
        self.vprof = None
        self._after_id = None
        self.horizon_pts = []
        self.target_pts = []

        # Variables
        self.p, self.w, self.c = {}, {}, {}
        self.model_type = tk.StringVar(value="Dynamic")
        self.use_bounds = tk.BooleanVar(value=True)
        self.show_horizon = tk.BooleanVar(value=True)
        self.show_targets = tk.BooleanVar(value=False)

        self._apply_styles()
        self._build_layout()
        self._init_simulation()

    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Dark.TNotebook", background=C["panel"], borderwidth=0, tabmargins=0)
        s.configure("Dark.TNotebook.Tab", background=C["tab_off"], foreground=C["muted"], padding=[10, 4], borderwidth=0)
        s.map("Dark.TNotebook.Tab", background=[("selected", C["tab_sel"])], foreground=[("selected", C["accent1"])])
        s.configure("Vert.TScrollbar", background=C["trough"], troughcolor=C["card"], borderwidth=0)

    def _build_layout(self):
        self.pane = tk.Frame(self.root, bg=C["panel"], width=300)
        self.pane.pack(side=tk.LEFT, fill=tk.Y)
        self.pane.pack_propagate(False)

        self.rframe = tk.Frame(self.root, bg=C["bg"])
        self.rframe.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Title & Telemetry
        bar = tk.Frame(self.rframe, bg=C["panel"], height=40)
        bar.pack(fill=tk.X)
        tk.Label(bar, text="◈  MPC UNIFIED CONTROLLER", bg=C["panel"], fg=C["accent1"], font=("Courier New", 11, "bold")).pack(side=tk.LEFT, padx=15)
        self.telemetry_var = tk.StringVar()
        tk.Label(bar, textvariable=self.telemetry_var, bg=C["panel"], fg=C["accent3"], font=("Courier New", 9)).pack(side=tk.RIGHT, padx=15)

        # Canvas
        self.fig = Figure(facecolor=C["bg"])
        self.ax = self.fig.add_subplot(111, facecolor=C["card"])
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05)
        self.ax.set_aspect("equal")
        self.ax.grid(True, color=C["border"], alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.rframe)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tabs
        nb = ttk.Notebook(self.pane, style="Dark.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        t_sim = self._make_scroll_tab(nb, "Sim")
        t_mpc = self._make_scroll_tab(nb, "MPC")
        t_track = self._make_scroll_tab(nb, "Track")
        t_car = self._make_scroll_tab(nb, "Car")

        self._fill_sim_tab(t_sim)
        self._fill_mpc_tab(t_mpc)
        self._fill_track_tab(t_track)
        self._fill_car_tab(t_car)

        # Playback
        row = tk.Frame(self.pane, bg=C["panel"])
        row.pack(fill=tk.X, padx=10, pady=10)
        self.btn_play = tk.Button(row, text="▶ Play", command=self._do_play, bg="#1a4a2a", fg="white", relief="flat", width=8, pady=5)
        self.btn_play.pack(side=tk.LEFT, expand=True, padx=2)
        tk.Button(row, text="⏸ Pause", command=self._do_pause, bg="#4a3a1a", fg="white", relief="flat", width=8, pady=5).pack(side=tk.LEFT, expand=True, padx=2)
        tk.Button(row, text="↺ Reset", command=self._do_reset, bg="#2a1a4a", fg="white", relief="flat", width=8, pady=5).pack(side=tk.LEFT, expand=True, padx=2)

    def _make_scroll_tab(self, nb, name):
        f = tk.Frame(nb, bg=C["panel"])
        c = tk.Canvas(f, bg=C["panel"], highlightthickness=0)
        s = ttk.Scrollbar(f, orient="vertical", command=c.yview)
        i = tk.Frame(c, bg=C["panel"])
        i.bind("<Configure>", lambda e: c.configure(scrollregion=c.bbox("all")))
        w = c.create_window((0,0), window=i, anchor="nw")
        c.configure(yscrollcommand=s.set)
        c.bind("<Configure>", lambda e: c.itemconfig(w, width=e.width))
        s.pack(side=tk.RIGHT, fill=tk.Y); c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        nb.add(f, text=f" {name} ")
        return i

    def _input(self, p, l, v, lo, hi, res=0.1):
        r = tk.Frame(p, bg=C["panel"]); r.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r, text=l, bg=C["panel"], fg=C["text"], font=("Helvetica", 9), anchor="w", width=18).pack(side=tk.LEFT)
        tk.Spinbox(r, from_=lo, to=hi, increment=res, textvariable=v, bg=C["slider"], fg=C["white"], buttonbackground=C["trough"], relief="flat", font=("Courier New", 9), width=8).pack(side=tk.RIGHT)

    def _section(self, p, t):
        tk.Frame(p, bg=C["border"], height=1).pack(fill=tk.X, padx=8, pady=(10,2))
        tk.Label(p, text=f"  {t}", bg=C["panel"], fg=C["accent1"], font=("Courier New", 8, "bold")).pack(fill=tk.X, padx=8, pady=(0,4))

    def _fill_sim_tab(self, p):
        self._section(p, "MODEL")
        cb = ttk.Combobox(p, textvariable=self.model_type, values=["Kinematic", "Dynamic"], state="readonly")
        cb.pack(fill=tk.X, padx=10, pady=5)
        
        self._section(p, "DYNAMICS")
        def s(l, k, lo, hi, r, i): self.p[k] = tk.DoubleVar(value=i); self._input(p, l, self.p[k], lo, hi, r)
        s("Time step dt", "dt", 0.01, 0.3, 0.01, 0.1)
        s("Max speed (m/s)", "max_speed", 1, 80, 1, 40)
        s("Min speed (m/s)", "min_speed", 0, 40, 1, 3)
        s("Max accel", "max_accel", 1, 25, 0.5, 9)
        s("Max decel", "max_decel", 1, 35, 0.5, 15)
        s("Max steer (°)", "max_steer", 5, 90, 1, 40)

        self._section(p, "NOISE")
        s("Accel noise σ", "noise_a", 0.0, 3.0, 0.05, 0.1)
        s("Steer noise σ (°)", "noise_steer_deg", 0.0, 10.0, 0.1, 1.0)
        
        self._section(p, "HORIZON")
        self.p["N_horizon"] = tk.IntVar(value=6); self._input(p, "Prediction N", self.p["N_horizon"], 2, 25, 1)
        s("Min lookahead (m)", "min_lookahead", 0, 15, 0.5, 3)

    def _fill_mpc_tab(self, p):
        self._section(p, "TRACKING")
        def s(l, k, lo, hi, r, i): self.w[k] = tk.DoubleVar(value=i); self._input(p, l, self.w[k], lo, hi, r)
        s("Q_dist (Pos)", "Q_dist", 0, 100, 1, 15)
        s("Q_psi (Hdg)", "Q_psi", 0, 100, 1, 10)
        s("Q_v (Vel)", "Q_v", 0, 100, 1, 5)
        s("Q_bound (Lim)", "Q_bound", 0, 1000, 10, 500)
        
        self._section(p, "EFFORT / SMOOTH")
        s("R_accel", "R_accel", 0, 5, 0.01, 0.01)
        s("R_steer", "R_steer", 0, 100, 1, 15)
        s("Rd_accel", "Rd_accel", 0, 5, 0.01, 0.01)
        s("Rd_steer", "Rd_steer", 0, 100, 1, 8)

    def _fill_track_tab(self, p):
        self._section(p, "SIZE & SHAPE")
        def s(l, k, lo, hi, r, i, t=tk.DoubleVar): self.p[k] = t(value=i); self._input(p, l, self.p[k], lo, hi, r)
        s("X Bounds", "x_size", 50, 1000, 10, 200)
        s("Y Bounds", "y_size", 50, 1000, 10, 200)
        s("Complexity", "n_control", 4, 40, 1, 12, tk.IntVar)
        s("Step Dist (m)", "step_dist", 0.5, 10, 0.5, 3.0)
        s("Track Width (m)", "track_width", 2, 30, 0.5, 8.0)
        s("Seed (-1 rand)", "track_seed", -1, 1000, 1, -1, tk.IntVar)
        
        self._section(p, "ACTIONS")
        tk.Checkbutton(p, text="Use Boundary Cost", variable=self.use_bounds, bg=C["panel"], fg=C["text"], selectcolor=C["card"], command=self._redraw_track).pack(anchor="w", padx=10)
        tk.Button(p, text="🔀 Generate New Track", command=self._do_gen_track, bg="#1a3a6a", fg="white", relief="flat", pady=5).pack(fill=tk.X, padx=10, pady=5)
        tk.Button(p, text="💾 Save Track", command=self._do_save_track, bg="#1a4a2a", fg="white", relief="flat", pady=3).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(p, text="📂 Load Track", command=self._do_load_track, bg="#3a2a1a", fg="white", relief="flat", pady=3).pack(fill=tk.X, padx=10, pady=2)

    def _fill_car_tab(self, p):
        self._section(p, "GEOMETRY")
        def s(l, k, lo, hi, r, i): self.c[k] = tk.DoubleVar(value=i); self._input(p, l, self.c[k], lo, hi, r)
        s("Wheelbase L", "L", 0.5, 5, 0.1, 2.5)
        s("Lf (CG→Front)", "Lf", 0.1, 4, 0.1, 1.2)
        s("Lr (CG→Rear)", "Lr", 0.1, 4, 0.1, 1.3)
        
        self._section(p, "INERTIA & TIRES")
        s("Mass (kg)", "m", 200, 5000, 50, 1500)
        s("Iz (kgm2)", "Iz", 200, 8000, 100, 3000)
        s("Ca_f (N/rad)", "Ca_f", 10000, 200000, 5000, 80000)
        s("Ca_r (N/rad)", "Ca_r", 10000, 200000, 5000, 80000)
        s("Friction mu", "mu", 0.1, 1.5, 0.05, 0.8)
        
        self._section(p, "VISUALS")
        tk.Checkbutton(p, text="Show Horizon", variable=self.show_horizon, bg=C["panel"], fg=C["text"], selectcolor=C["card"]).pack(anchor="w", padx=10)
        tk.Checkbutton(p, text="Show Targets", variable=self.show_targets, bg=C["panel"], fg=C["text"], selectcolor=C["card"]).pack(anchor="w", padx=10)
        tk.Button(p, text="↺ Apply & Reset", command=self._do_reset, bg="#2a1a4a", fg="white", relief="flat", pady=5).pack(fill=tk.X, padx=10, pady=10)

    # Logic functions
    def _init_simulation(self):
        if _HAS_EXT_TRACK:
            try:
                self.track = generate_track_for_mpc(
                    num_points=self.p["n_control"].get(),
                    x_bounds=[0, self.p["x_size"].get()],
                    y_bounds=[0, self.p["y_size"].get()],
                    step_distance=self.p["step_dist"].get()
                )
            except: self.track = _builtin_track(self.p["step_dist"].get(), self.p["n_control"].get(), self.p["track_seed"].get(), self.p["x_size"].get(), self.p["y_size"].get())
        else:
            self.track = _builtin_track(self.p["step_dist"].get(), self.p["n_control"].get(), self.p["track_seed"].get(), self.p["x_size"].get(), self.p["y_size"].get())
        
        self.left_track, self.right_track = generate_boundaries(self.track, self.p["track_width"].get())
        self._reset_state()
        self._redraw_track()

    def _reset_state(self):
        self.frame_num = 0
        self.hist_x.clear(); self.hist_y.clear()
        self.last_u = np.array([0.0, 0.0])
        if self.track is None: return
        
        N = self.p["N_horizon"].get()
        self.u_guess = np.zeros(2 * N)
        self.vprof = get_speed_profile(self.track, self.p["max_speed"].get(), self.p["min_speed"].get())
        
        x0, y0 = self.track[0]
        th0 = np.arctan2(self.track[1,1]-y0, self.track[1,0]-x0)
        
        params = {**{k: v.get() for k,v in self.p.items()}, **{k: v.get() for k,v in self.c.items()}}
        if self.model_type.get() == "Dynamic":
            self.car = CarDynamics4Wheel(x=x0, y=y0, theta=th0, **params)
            # Match stiffness if in car_model
            self.car.Ca_f = self.c["Ca_f"].get()
            self.car.Ca_r = self.c["Ca_r"].get()
        else:
            self.car = KinematicBicycleModel(x=x0, y=y0, theta=th0, **params)

    def _redraw_track(self):
        self.ax.clear()
        if self.track is not None:
            t_c = np.vstack([self.track, self.track[0]])
            self.ax.plot(t_c[:,0], t_c[:,1], "--", color=C["muted"], alpha=0.3)
            if self.use_bounds.get():
                l_c = np.vstack([self.left_track, self.left_track[0]])
                r_c = np.vstack([self.right_track, self.right_track[0]])
                self.ax.plot(l_c[:,0], l_c[:,1], "-", color=C["muted"], linewidth=1.5, alpha=0.6)
                self.ax.plot(r_c[:,0], r_c[:,1], "-", color=C["muted"], linewidth=1.5, alpha=0.6)
            self.ax.scatter(self.track[:,0], self.track[:,1], c=self.vprof, cmap="RdYlGn", s=5, alpha=0.5)
        
        self.path_line, = self.ax.plot([], [], "-", color=C["accent1"], linewidth=1.5, alpha=0.8)
        self.car_body = patches.Polygon(np.zeros((4,2)), closed=True, edgecolor=C["accent3"], facecolor=C["accent3"], alpha=0.8, zorder=10)
        self.ax.add_patch(self.car_body)
        self.horizon_line, = self.ax.plot([], [], "o-", color=C["accent2"], markersize=3, linewidth=1, alpha=0.7)
        self.target_scatter = self.ax.scatter([], [], marker="x", color="#bf5fff", s=40, zorder=11)
        self.canvas.draw()

    def _step(self):
        if not self.running: return
        N, dt = self.p["N_horizon"].get(), self.p["dt"].get()
        if len(self.u_guess) != 2 * N: self.u_guess = np.zeros(2 * N)
        
        bounds = [(-self.p["max_decel"].get(), self.p["max_accel"].get()), (-np.radians(self.p["max_steer"].get()), np.radians(self.p["max_steer"].get()))] * N
        weights = (self.w["Q_dist"].get(), self.w["Q_psi"].get(), self.w["Q_v"].get(), self.w["R_accel"].get(), self.w["R_steer"].get(), self.w["Rd_accel"].get(), self.w["Rd_steer"].get(), self.w["Q_bound"].get())
        params = {**{k: v.get() for k,v in self.p.items()}, **{k: v.get() for k,v in self.c.items()}}
        params['max_steer'] = np.radians(params['max_steer'])
        
        # Update simulation model parameters live
        for k,v in params.items():
            if hasattr(self.car, k): setattr(self.car, k, v)
        if self.model_type.get() == "Dynamic":
            self.car.Ca_f, self.car.Ca_r = self.c["Ca_f"].get(), self.c["Ca_r"].get()

        res = minimize(mpc_cost, self.u_guess, args=(self.car.get_state(), self.model_type.get(), params, self.track, self.left_track, self.right_track, self.use_bounds.get(), N, dt, self.last_u, self.vprof, weights), method="SLSQP", bounds=bounds, options={"ftol": 1e-2, "maxiter": 30})
        
        a_opt, d_opt = res.x[:2]
        self.car.update(a_opt, d_opt, dt)
        self.u_guess = np.roll(res.x, -2); self.u_guess[-2:] = res.x[-2:]
        self.last_u = np.array([a_opt, d_opt])

        # Viz updates
        x, y, th, v, w = self.car.get_state()
        self.hist_x.append(x); self.hist_y.append(y)
        self.path_line.set_data(self.hist_x, self.hist_y)
        
        # Car body
        L = self.c["L"].get(); W = L * 0.5
        Lf, Lr = self.c["Lf"].get(), self.c["Lr"].get()
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        corners = np.array([[Lf, W/2], [Lf, -W/2], [-Lr, -W/2], [-Lr, W/2]]) @ R.T + [x, y]
        self.car_body.set_xy(corners)
        
        # Horizon & Targets
        if self.show_horizon.get():
            h_pts = [[x, y]]
            hx, hy, hth, hv, hw = x, y, th, v, w
            for i in range(N):
                ha, hd = res.x[2*i:2*i+2]
                hv = np.clip(hv + ha*dt, 0.1, params['max_speed']*1.1)
                if self.model_type.get() == "Dynamic":
                    # Simple approx for viz
                    hth += hw*dt; hx += hv*np.cos(hth)*dt; hy += hv*np.sin(hth)*dt
                else:
                    hx += hv*np.cos(hth)*dt; hy += hv*np.sin(hth)*dt; hth += (hv/L)*np.tan(hd)*dt
                h_pts.append([hx, hy])
            self.horizon_line.set_data(np.array(h_pts)[:,0], np.array(h_pts)[:,1])
        else: self.horizon_line.set_data([], [])

        if self.show_targets.get():
            t_pts = []
            dists = np.sum((self.track - np.array([x, y]))**2, axis=1)
            idx = np.argmin(dists)
            dist_per_idx = np.linalg.norm(self.track[1]-self.track[0])
            curr_v = v
            for i in range(N):
                curr_v = np.clip(curr_v + res.x[2*i]*dt, 0.1, params['max_speed'])
                la = max(params['min_lookahead'], curr_v * dt * (i+1))
                ti = (idx + int(round(la/dist_per_idx))) % len(self.track)
                t_pts.append(self.track[ti])
            self.target_scatter.set_offsets(t_pts)
        else: self.target_scatter.set_offsets(np.empty((0,2)))

        self.telemetry_var.set(f"V: {v:.1f} m/s | A: {a_opt:+.1f} | Δ: {np.degrees(d_opt):+.1f}° | Cost: {res.fun:.0f}")
        self.canvas.draw_idle()
        self._after_id = self.root.after(int(dt*1000), self._step)

    def _do_play(self): self.running = True; self._step()
    def _do_pause(self): self.running = False
    def _do_reset(self): self._do_pause(); self._reset_state(); self._redraw_track()
    def _do_gen_track(self): self._do_pause(); self._init_simulation()
    
    def _do_save_track(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv")
        if p: np.savetxt(p, self.track, delimiter=",", header="x,y", comments="")
    
    def _do_load_track(self):
        p = filedialog.askopenfilename()
        if p:
            self.track = np.loadtxt(p, delimiter=",", skiprows=1)
            self.left_track, self.right_track = generate_boundaries(self.track, self.p["track_width"].get())
            self._do_reset()

if __name__ == "__main__":
    root = tk.Tk()
    app = MPCCarApp(root)
    root.mainloop()
