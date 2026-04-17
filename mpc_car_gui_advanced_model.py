"""
MPC Car Controller — Interactive GUI
─────────────────────────────────────
Controls: start/stop/reset, live parameter tuning, track generation,
          save/load tracks (CSV or .npy).

Dependencies:  numpy, scipy, matplotlib, tkinter (stdlib)
Optional:      random_racetrack_mpc  (falls back to built-in generator)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib import path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib import patches
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from car_model import CarDynamics4Wheel, pacejka

# ── Try external track generator ──────────────────────────────────────────────
try:
    from random_racetrack_mpc import generate_track_for_mpc as _ext_gen_track
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False


# ═══════════════════════════════════════════════════════════════════════════════
#  TRACK GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _resample(points, step_distance):
    """Resample a 2-D path so every consecutive pair is ~step_distance apart."""
    diffs = np.diff(points, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.insert(np.cumsum(seg_len), 0, 0)
    total = cum[-1]
    ds = np.arange(0, total, step_distance)
    fx = interp1d(cum, points[:, 0], kind="linear")
    fy = interp1d(cum, points[:, 1], kind="linear")
    return np.vstack([fx(ds), fy(ds)]).T


def _builtin_track(step_distance=3.0, n_control=12, seed=None):
    """Procedural closed racetrack using polar-jittered control points."""
    rng = np.random.default_rng(seed if seed >= 0 else None)
    angles = np.linspace(0, 2 * np.pi, n_control, endpoint=False)
    radii = rng.uniform(35, 75, n_control)
    jitter = rng.uniform(-0.25, 0.25, n_control)
    cx = radii * np.cos(angles + jitter) + 100
    cy = radii * np.sin(angles + jitter) + 100
    # Close loop
    cx = np.append(cx, cx[0])
    cy = np.append(cy, cy[0])
    t = np.linspace(0, 1, len(cx))
    t_fine = np.linspace(0, 1, 3000)
    fine = np.vstack([
        interp1d(t, cx, kind="cubic")(t_fine),
        interp1d(t, cy, kind="cubic")(t_fine),
    ]).T
    return _resample(fine, step_distance)


def generate_track(step_distance=3.0, n_control=12, seed=-1):
    if _HAS_EXT:
        try:
            return _ext_gen_track(step_distance=step_distance)
        except Exception:
            pass
    return _builtin_track(step_distance, n_control, seed)


# ═══════════════════════════════════════════════════════════════════════════════
#  MPC
# ═══════════════════════════════════════════════════════════════════════════════

def speed_profile(track, max_speed, min_speed=3.0, curvature_gain=5.0,
                 smoothing_window=5, lookahead=10):
    """
    Generate a speed profile based on track curvature with proper error handling.
    """
    # 1. Better curvature calculation using 3-point method
    curvatures = np.zeros(len(track))

    # Handle edge cases for short tracks
    if len(track) < 3:
        return np.full(len(track), min_speed)

    for i in range(1, len(track)-1):
        p0 = track[i-1]
        p1 = track[i]
        p2 = track[i+1]

        # Calculate vectors
        v1 = p1 - p0
        v2 = p2 - p1

        # Normalize with safety checks
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            curvatures[i] = 0.0
            continue

        v1 = v1 / norm1
        v2 = v2 / norm2

        # Curvature calculation
        cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        d = np.linalg.norm(p2 - p0)/2
        curvatures[i] = 2 * np.sin(theta/2) / (d + 1e-6) if d > 1e-6 else 0.0

    # Handle endpoints by copying nearby values
    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]

    # 2. Apply lookahead safely
    extended_curv = np.copy(curvatures)
    for i in range(len(track)):
        future_indices = range(i, min(i + lookahead + 1, len(track)))
        if len(future_indices) > 0:
            extended_curv[i] = max(curvatures[future_indices])
        else:
            extended_curv[i] = curvatures[i]

    # 3. Calculate base speeds with safety check
    v = np.zeros(len(track))
    for i in range(len(track)):
        v[i] = max_speed / (1 + curvature_gain * extended_curv[i])

    # 4. Apply smoothing safely
    if smoothing_window > 1 and len(v) > smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        v = np.convolve(v, kernel, mode='same')
    elif smoothing_window > 1:
        # If track is shorter than window, just average all
        v[:] = np.mean(v)

    # 5. Apply min/max limits
    return np.clip(v, min_speed, max_speed)


def mpc_cost(u_flat, state, L, Lf, Lr, track, N, dt, last_u, vprof, weights, max_accel, max_decel, max_steer, max_speed, Iz=3000.0, Ca_f=80000.0, Ca_r=80000.0, min_lookahead=3.0):
    Q_dist, Q_psi, Q_v, R_a, R_s, Rd_a, Rd_s = weights
    controls = u_flat.reshape(-1, 2)
    x, y, th, v, w = state  # Updated to include yaw rate

    # Find nearest point on track
    dists = np.sum((track - np.array([x, y])) ** 2, axis=1)
    idx = int(np.argmin(dists))

    total = 0.0
    pa, pd = last_u

    # Get current slip angles for more accurate prediction
    beta = np.arctan2(Lr * w, v) if v > 0.1 else 0.0
    
    p1 = track[idx]
    p2 = track[(idx + 1) % len(track)]
    dist_per_index = np.linalg.norm(p2 - p1)

    # Ensure we don't divide by zero
    dist_per_index = max(dist_per_index, 0.1)

    for i in range(N):
        a, delta = controls[i]

        # 1. CLIP FIRST
        a = np.clip(a, -max_decel, max_accel)  # Use your decel/accel parameters here
        delta = np.clip(delta, -max_steer, max_steer) # Use your max steer in radians here

        # 2. UPDATE v_new and CLIP IT IMMEDIATELY
        v_new = np.clip(v + a * dt, 0.1, max_speed + max_speed/10) # Use your min/max speed parameters here, and ensure a small positive lower bound to avoid issues with slip angle calculation

        # 3. NOW calculate forces using the clipped velocity
        alpha_f = delta - beta
        alpha_r = -np.arctan2(Lr * w, v_new) 
        
        Fy_f = -Ca_f * alpha_f
        Fy_r = -Ca_r * alpha_r

        w_new = w + (Lf * Fy_f * np.cos(delta) - Lr * Fy_r) / Iz * dt
        th_new = th + w * dt
        x_new = x + v_new * np.cos(th) * dt
        y_new = y + v_new * np.sin(th) * dt

        # Update slip angle for next iteration
        beta = np.arctan2(Lr * w_new, v_new) if v_new > 0.1 else 0.0

        base_speed = 30.0  # Speed where weights = base values
        Rs_dyn = R_s * max(1.0, (v_new/base_speed)**2)  # Never less than R_s
        Rds_dyn = Rd_s * max(1.0, (v_new/base_speed)**2)

        # Get reference point on track and scale in function of speed and index to prevent "skipping" at high speeds
        # Calculate the physical distance the car will have traveled at this horizon step
        # We use (i + 1) because we are looking at the state AFTER dt
        # 1. Define your minimum look-ahead distance (in meters)
        # For example, at least 5 meters ahead regardless of speed
        min_lookahead = min_lookahead  # This can be tuned or even made a function of speed if desired

        # 2. Calculate the distance based on speed
        speed_based_distance = v_new * (dt * (i + 1))

        # 3. Take the larger of the two
        # This ensures a 'Minimum Horizon' of indices
        predicted_distance = max(min_lookahead, speed_based_distance)
        
   
        idx_offset = int(round(predicted_distance / dist_per_index))
        ri = (idx + idx_offset) % len(track)
        ni = (ri + 1) % len(track)
        ref = track[ri]
        target_th = np.arctan2(track[ni, 1] - ref[1], track[ni, 0] - ref[0])

        # Calculate errors
        dsq = (x_new - ref[0]) ** 2 + (y_new - ref[1]) ** 2  # Distance error
        herr = (th_new - target_th + np.pi) % (2 * np.pi) - np.pi  # Heading error
        verr = (v_new - vprof[ri]) ** 2  # Velocity error

        # Add to cost
        total += Q_dist * dsq + Q_psi * herr ** 2 + Q_v * verr
        total += R_a * a ** 2 + Rs_dyn * delta ** 2
        total += Rd_a * (a - pa) ** 2 + Rds_dyn * (delta - pd) ** 2
        
        # Coupling penalty: punishes high acceleration AND high steering simultaneously
        total += 0 * (a**2 * delta**2)

        # Update state for next iteration
        x, y, th, v, w, pa, pd = x_new, y_new, th_new, v_new, w_new, a, delta

    return total



# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

C = {
    "bg":       "#0d0d14",
    "panel":    "#13131f",
    "card":     "#1a1a2e",
    "border":   "#252542",
    "accent1":  "#00e5ff",   # cyan
    "accent2":  "#ff4d6d",   # red
    "accent3":  "#b5ff4d",   # green
    "text":     "#c8c8e0",
    "muted":    "#5a5a88",
    "white":    "#f0f0ff",
    "slider":   "#1e1e38",
    "trough":   "#2a2a48",
    "tab_sel":  "#20203a",
    "tab_off":  "#16162a",
}

FONT_MONO  = ("Courier New", 9)
FONT_LABEL = ("Helvetica",   9)
FONT_HEAD  = ("Helvetica",  10, "bold")
FONT_TITLE = ("Helvetica",  13, "bold")


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class MPCCarApp:
    # ── init ─────────────────────────────────────────────────────────────────

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MPC Car Controller")
        self.root.configure(bg=C["bg"])
        self.root.minsize(1000, 680)

        # Runtime state
        self.running    = False
        self.frame_num  = 0
        self.hist_x: list[float] = []
        self.hist_y: list[float] = []
        self.track: np.ndarray | None = None
        self.car: CarDynamics4Wheel | None = None
        self.u_guess    = np.zeros(10)
        self.last_u     = np.array([0.0, 0.0])
        self.vprof: np.ndarray | None = None
        self._after_id  = None
        self.show_horizon = tk.BooleanVar(value=False)
        self.horizon_pts: list = []
        self.show_targets = tk.BooleanVar(value=False)
        self.target_pts: list = []

        # Param variables (filled in _build_controls)
        self.p: dict[str, tk.Variable] = {}
        self.w: dict[str, tk.Variable] = {}
        self.c: dict[str, tk.Variable] = {}  # car model params

        self._apply_styles()
        self._build_layout()
        self._init_simulation()

    # ── ttk styling ──────────────────────────────────────────────────────────

    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use("clam")

        s.configure("Dark.TNotebook",
                    background=C["panel"], borderwidth=0, tabmargins=0)
        s.configure("Dark.TNotebook.Tab",
                    background=C["tab_off"], foreground=C["muted"],
                    padding=[12, 5], font=FONT_LABEL, borderwidth=0)
        s.map("Dark.TNotebook.Tab",
              background=[("selected", C["tab_sel"])],
              foreground=[("selected", C["accent1"])])

        s.configure("Vert.TScrollbar",
                    background=C["trough"], troughcolor=C["card"],
                    arrowcolor=C["muted"], borderwidth=0)

    # ── layout skeleton ───────────────────────────────────────────────────────

    def _build_layout(self):
        # Left control panel
        self.pane = tk.Frame(self.root, bg=C["panel"], width=272)
        self.pane.pack(side=tk.LEFT, fill=tk.Y)
        self.pane.pack_propagate(False)

        # Right — title bar + canvas
        self.rframe = tk.Frame(self.root, bg=C["bg"])
        self.rframe.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_title_bar()
        self._build_canvas()
        self._build_controls()

    def _build_title_bar(self):
        bar = tk.Frame(self.rframe, bg=C["panel"], height=38)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        tk.Label(bar, text="◈  MPC CAR CONTROLLER",
                 bg=C["panel"], fg=C["accent1"],
                 font=("Courier New", 11, "bold")).pack(side=tk.LEFT, padx=14)

        self.telemetry_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.telemetry_var,
                 bg=C["panel"], fg=C["accent3"],
                 font=FONT_MONO).pack(side=tk.RIGHT, padx=14)

    # ── matplotlib canvas ─────────────────────────────────────────────────────

    def _build_canvas(self):
        self.fig = Figure(facecolor=C["bg"])
        self.ax  = self.fig.add_subplot(111, facecolor=C["card"])
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.07)
        self._style_ax()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.rframe)
        wid = self.canvas.get_tk_widget()
        wid.configure(bg=C["bg"], highlightthickness=0)
        wid.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor(C["card"])
        ax.tick_params(colors=C["muted"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", color=C["muted"], fontsize=8)
        ax.set_ylabel("Y (m)", color=C["muted"], fontsize=8)
        ax.grid(True, color=C["border"], linewidth=0.5, alpha=0.6)

    # ── left panel ────────────────────────────────────────────────────────────

    def _build_controls(self):
        # Panel header
        hdr = tk.Frame(self.pane, bg=C["panel"])
        hdr.pack(fill=tk.X, padx=10, pady=(10, 2))
        tk.Label(hdr, text="PARAMETERS", bg=C["panel"],
                 fg=C["accent1"], font=("Courier New", 10, "bold")).pack(side=tk.LEFT)

        # Notebook tabs
        nb = ttk.Notebook(self.pane, style="Dark.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        t_sim   = self._make_scroll_tab(nb, "Sim")
        t_mpc   = self._make_scroll_tab(nb, "MPC")
        t_track = self._make_scroll_tab(nb, "Track")
        t_car   = self._make_scroll_tab(nb, "Car")

        nb.add(t_sim["outer"],   text=" ⚙ Sim ")
        nb.add(t_mpc["outer"],   text=" 🎯 MPC ")
        nb.add(t_track["outer"], text=" 🗺 Track")
        nb.add(t_car["outer"],   text=" 🚗 Car ")

        self._fill_sim_tab(t_sim["inner"])
        self._fill_mpc_tab(t_mpc["inner"])
        self._fill_track_tab(t_track["inner"])
        self._fill_car_tab(t_car["inner"])

        # Separator
        tk.Frame(self.pane, bg=C["border"], height=1).pack(fill=tk.X, padx=8, pady=6)

        # Playback buttons
        self._build_playback()

        # Status readout
        self.status_var = tk.StringVar(value="Idle — press ▶ Play to start")
        st = tk.Label(self.pane, textvariable=self.status_var,
                      bg=C["panel"], fg=C["text"],
                      font=FONT_MONO, anchor="w", justify=tk.LEFT,
                      wraplength=255, pady=4)
        st.pack(fill=tk.X, padx=10, pady=(0, 6))

    # ── scrollable tab factory ────────────────────────────────────────────────

    def _make_scroll_tab(self, nb, name):
        outer  = tk.Frame(nb, bg=C["panel"])
        canvas = tk.Canvas(outer, bg=C["panel"], highlightthickness=0)
        # Just use the default themed scrollbar, which will still respect 'clam'
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner  = tk.Frame(canvas, bg=C["panel"])
        inner.bind("<Configure>",
                   lambda e: canvas.configure(
                       scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)

        # Make inner fill the canvas width
        def _on_canvas_resize(e):
            canvas.itemconfig(win, width=e.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mouse-wheel anywhere inside the tab
        def _scroll(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        outer.bind_all("<MouseWheel>", _scroll)

        return {"outer": outer, "inner": inner}

    # ── widget helpers ────────────────────────────────────────────────────────

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=C["border"], height=1)
        f.pack(fill=tk.X, padx=8, pady=(10, 2))
        tk.Label(parent, text=f"  {title}",
                 bg=C["panel"], fg=C["accent1"],
                 font=("Courier New", 8, "bold")).pack(fill=tk.X, padx=8, pady=(0, 4))

    def _number_input(self, parent, label, var, lo, hi, res=None, fmt=None):
        """Label row + Spinbox for precise number entry."""
        res = res or 0.1
        row = tk.Frame(parent, bg=C["panel"])
        row.pack(fill=tk.X, padx=8, pady=4)

        # Label
        tk.Label(row, text=label, bg=C["panel"], fg=C["text"],
                 font=FONT_LABEL, anchor="w", width=22,
                 wraplength=155).pack(side=tk.LEFT)

        # Spinbox (Replaces the Scale slider)
        # 'from_' and 'to' define the range, 'increment' is the step
        sp = tk.Spinbox(row, from_=lo, to=hi, increment=res,
                        textvariable=var,
                        bg=C["slider"], fg=C["white"], 
                        buttonbackground=C["trough"],
                        insertbackground=C["white"],
                        relief="flat", font=FONT_MONO, 
                        justify="center", width=12)
        sp.pack(side=tk.RIGHT, padx=2)
        
        return sp

    def _btn(self, parent, text, cmd, color=None, pady=4):
        color = color or C["card"]
        b = tk.Button(parent, text=text, command=cmd,
                      bg=color, fg=C["white"],
                      activebackground=C["accent1"], activeforeground=C["bg"],
                      relief="flat", bd=0, cursor="hand2",
                      font=FONT_HEAD, pady=pady)
        b.pack(fill=tk.X, padx=8, pady=2)
        return b

    # ── Sim tab ───────────────────────────────────────────────────────────────

    def _fill_sim_tab(self, p):
        def s(label, key, lo, hi, res=None, init=None):
            self.p[key] = tk.DoubleVar(value=init or lo)
            self._number_input(p, label, self.p[key], lo, hi, res)

        def si(label, key, lo, hi, res=1, init=None):
            self.p[key] = tk.IntVar(value=init or lo)
            self._number_input(p, label, self.p[key], lo, hi, res)

        self._section(p, "DYNAMICS")
        s("Time step dt  (s)",       "dt",         0.01, 0.30, 0.01, 0.10)
        s("Max speed  (m/s)",        "max_speed",   1.0, 80.0,  1.0, 20.0)
        s("Min speed  (m/s)",        "min_speed",   0.0, 40.0,  1.0,  3.0)
        s("Max accel  (m/s²)",       "max_accel",   1.0, 25.0,  0.5,  9.0)
        s("Max decel  (m/s²)",       "max_decel",   1.0, 35.0,  0.5, 15.0)
        s("Max steer  (°)",          "max_steer",   5.0, 90.0,  1.0, 40.0)
        s("Wheelbase L  (m)",        "wheelbase",   0.5,  5.0,  0.1,  2.5)

        self._section(p, "NOISE")
        s("Accel noise σ",           "noise_a",     0.0,  3.0, 0.05, 0.20)
        s("Steer noise σ  (°)",      "noise_steer", 0.0, 10.0, 0.10, 1.00)

        self._section(p, "HORIZON")
        si("Minimum look-ahead (m)", "min_lookahead", 0, 10, 0.5, 3)
        si("Prediction horizon N",   "N_horizon",   2, 20,  1,  5)

    # ── MPC weights tab ───────────────────────────────────────────────────────

    def _fill_mpc_tab(self, p):
        def s(label, key, lo, hi, res, init):
            self.w[key] = tk.DoubleVar(value=init)
            self._number_input(p, label, self.w[key], lo, hi, res)

        self._section(p, "TRACKING COST")
        s("Q_dist   position",       "Q_dist",   0.0,  80.0,  0.5, 10.0)
        s("Q_psi    heading",        "Q_psi",    0.0,  80.0,  0.5, 5.0)
        s("Q_v      velocity",       "Q_v",      0.0,  40.0,  0.5,  15.0)

        self._section(p, "EFFORT COST")
        s("R_accel  accel",          "R_accel",  0.0,   2.0, 0.01, 0.1)
        s("R_steer  steering",       "R_steer",  0.0,  80.0,  0.5, 15.0)

        self._section(p, "SMOOTHNESS COST")
        s("Rd_accel rate of accel",  "Rd_accel", 0.0,   2.0, 0.01, 0.1)
        s("Rd_steer rate of steer",  "Rd_steer", 0.0,  50.0,  0.5,  8.0)

    # ── Track tab ─────────────────────────────────────────────────────────────

    def _fill_track_tab(self, p):
        self._section(p, "GENERATION")

        self.p["step_dist"]  = tk.DoubleVar(value=3.0)
        self.p["n_control"]  = tk.IntVar(value=12)
        self.p["track_seed"] = tk.IntVar(value=-1)

        self._number_input(p, "Step distance (m)",    self.p["step_dist"],  0.5, 10.0, 0.5)
        self._number_input(p, "Complexity (pts)",     self.p["n_control"],  4,   25,   1)
        self._number_input(p, "Seed  (-1 = random)",  self.p["track_seed"], -1,  200,  1)

        self._section(p, "ACTIONS")
        self._btn(p, "🔀  Generate New Track", self._do_gen_track,   "#1a3a6a")
        self._btn(p, "💾  Save Track (CSV)",   self._do_save_track,  "#1a4a2a")
        self._btn(p, "📂  Load Track (CSV)",   self._do_load_track,  "#3a2a1a")
        self._btn(p, "📂  Load MIDPOINTS (CSV)",   self._do_load_midpoints,  "#3a2a1a")

    # ── Car model tab ─────────────────────────────────────────────────────────

    def _fill_car_tab(self, p):
        def s(label, key, lo, hi, res=None, init=None):
            self.c[key] = tk.DoubleVar(value=init if init is not None else lo)
            self._number_input(p, label, self.c[key], lo, hi, res)

        self._section(p, "GEOMETRY")
        s("Lf  CG→front axle (m)",   "Lf",   0.1,  4.0,  0.05, 1.2)
        s("Lr  CG→rear axle  (m)",   "Lr",   0.1,  4.0,  0.05, 1.3)

        self._section(p, "INERTIA")
        s("Mass  m  (kg)",            "m",    200, 5000,  50,  1500)
        s("Yaw inertia  Iz (kg·m²)",  "Iz",   200, 8000, 100,  3000)

        self._section(p, "TYRE STIFFNESS")
        s("Ca_f  front  (N/rad)",     "Ca_f", 10000, 200000, 5000, 80000)
        s("Ca_r  rear   (N/rad)",     "Ca_r", 10000, 200000, 5000, 80000)
        s("Friction  μ",              "mu",   0.1,   1.5,   0.05,  0.8)

        self._section(p, "VISUALISATION")
        tk.Checkbutton(
            p, text=" Show MPC Horizon",
            variable=self.show_horizon,
            bg=C["panel"], fg=C["text"],
            activebackground=C["panel"], activeforeground=C["accent1"],
            selectcolor=C["card"],
            font=FONT_LABEL, anchor="w",
            command=self._on_horizon_toggle,
        ).pack(fill=tk.X, padx=10, pady=4)

        tk.Checkbutton(
            p, text=" Show Target Points",
            variable=self.show_targets,
            bg=C["panel"], fg=C["text"],
            activebackground=C["panel"], activeforeground=C["accent1"],
            selectcolor=C["card"],
            font=FONT_LABEL, anchor="w",
            command=self._on_targets_toggle,
        ).pack(fill=tk.X, padx=10, pady=4)

        self._btn(p, "↺  Apply & Reset Car", self._do_apply_car, "#2a1a4a")

    # ── Playback buttons ──────────────────────────────────────────────────────

    def _on_horizon_toggle(self):
        """Show or hide the horizon line immediately."""
        if not self.show_horizon.get():
            self.horizon_line.set_data([], [])
            self.horizon_scatter.set_offsets(np.empty((0, 2)))
            self.canvas.draw_idle()

    def _on_targets_toggle(self):
        """Show or hide the MPC target points immediately."""
        if not self.show_targets.get():
            self.target_scatter.set_offsets(np.empty((0, 2)))
            self.canvas.draw_idle()

    def _do_apply_car(self):
        """Apply car-model params and hard-reset the simulation."""
        was = self.running
        self._do_pause()
        self._reset_state()
        self._redraw_track()
        self.status_var.set("Car model updated — press ▶ Play")
        if was:
            self._do_play()

    def _build_playback(self):
        row = tk.Frame(self.pane, bg=C["panel"])
        row.pack(fill=tk.X, padx=8, pady=4)
        for col in range(3):
            row.columnconfigure(col, weight=1)

        def pb(text, cmd, col, color):
            b = tk.Button(row, text=text, command=cmd,
                          bg=color, fg=C["white"],
                          activebackground=C["accent1"], activeforeground=C["bg"],
                          relief="flat", bd=0, cursor="hand2",
                          font=("Helvetica", 11, "bold"), pady=7)
            b.grid(row=0, column=col, padx=2, sticky="ew")
            return b

        self.btn_play  = pb("▶ Play",  self._do_play,  0, "#1a4a2a")
        self.btn_pause = pb("⏸ Pause", self._do_pause, 1, "#4a3a1a")
        pb("↺ Reset",                  self._do_reset, 2, "#2a1a4a")

    # ═════════════════════════════════════════════════════════════════════════
    #  SIMULATION LOGIC
    # ═════════════════════════════════════════════════════════════════════════

    def _init_simulation(self):
        self.track = generate_track(
            step_distance=self.p["step_dist"].get(),
            n_control=self.p["n_control"].get(),
            seed=int(self.p["track_seed"].get()),
        )
        self._reset_state()
        self._redraw_track()

    def _reset_state(self):
        self.frame_num = 0
        self.hist_x.clear()
        self.hist_y.clear()
        self.last_u = np.array([0.0, 0.0])

        if self.track is None:
            return

        N = self.p["N_horizon"].get()
        self.u_guess = np.zeros(2 * N)
        self.vprof   = speed_profile(self.track, self.p["max_speed"].get(), self.p["min_speed"].get())

        x0, y0 = self.track[0]
        th0 = np.arctan2(
            self.track[1, 1] - self.track[0, 1],
            self.track[1, 0] - self.track[0, 0],
        )
        # Pull car-model params (may not exist yet on first call)
        Lf = self.c["Lf"].get() if self.c else 1.2
        Lr = self.c["Lr"].get() if self.c else 1.3
        m  = self.c["m"].get()  if self.c else 1500
        Iz = self.c["Iz"].get() if self.c else 3000
        mu = self.c["mu"].get() if self.c else 0.8

        self.car = CarDynamics4Wheel(
            x=x0, y=y0, theta=th0,
            L=self.p["wheelbase"].get(),
            Lf=Lf, Lr=Lr, m=m, Iz=Iz, mu=mu,
            noise_a=self.p["noise_a"].get(),
            noise_steer_deg=self.p["noise_steer"].get(),
            max_accel=self.p["max_accel"].get(),
            max_decel=self.p["max_decel"].get(),
            max_steer=self.p["max_steer"].get(),
            max_speed=self.p["max_speed"].get()
        )

    # ── plot ──────────────────────────────────────────────────────────────────

    def _redraw_track(self):
        self.ax.clear()
        self._style_ax()

        if self.track is not None and len(self.track):
            t_closed = np.vstack([self.track, self.track[0]])
            # Speed-coloured scatter underneath
            vp = speed_profile(self.track, self.p["max_speed"].get(), self.p["min_speed"].get())
            self.ax.scatter(self.track[:, 0], self.track[:, 1],
                            c=vp, cmap="RdYlGn", s=8, alpha=0.55, zorder=2,
                            vmin=vp.min(), vmax=vp.max())
            # Dashed centreline
            self.ax.plot(t_closed[:, 0], t_closed[:, 1],
                         "--", color=C["accent2"],
                         linewidth=1.0, alpha=0.55, zorder=3, label="Ref")

        # Driven-path + car markers (will be updated each frame)
        self.path_line, = self.ax.plot(
            [], [], "-", color=C["accent1"], linewidth=1.8, alpha=0.85, zorder=4,
            label="Driven path")
        
        # Initialize with a dummy polygon (will be updated immediately)
        self.car_polygon = patches.Polygon(
            np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),  # Dummy coordinates
            closed=True,
            edgecolor=C["accent3"],
            facecolor=C["accent3"],
            linewidth=1,
            zorder=6,
            label="Car"
        )
        self.ax.add_patch(self.car_polygon)
        
        self.heading_line, = self.ax.plot(
            [], [], "-", color=C["accent3"], linewidth=1.5, alpha=0.7, zorder=5)

        # MPC horizon (toggleable)
        self.horizon_line, = self.ax.plot(
            [], [], "--", color="#ff9f1c", linewidth=1.4, alpha=0.85,
            zorder=7, label="MPC horizon")
        self.horizon_scatter = self.ax.scatter(
            [], [], c="#ff9f1c", s=12, alpha=0.7, zorder=8)

        # MPC target points (toggleable)
        self.target_scatter = self.ax.scatter(
            [], [], c="#bf5fff", s=45, alpha=0.90, zorder=9,
            marker="x", linewidths=1.8, label="MPC targets")

        leg = self.ax.legend(facecolor=C["card"], edgecolor=C["border"],
                             labelcolor=C["text"], fontsize=8, loc="upper right")
        self.canvas.draw()
    
    def update_car(self, x, y, theta, L=4.5, Lf=1.2, Lr=1.8):
        """Update the car's position and orientation"""
        car_width = L * 0.6

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

        # Update polygon vertices
        self.car_polygon.set_xy(corners)

    def _update_plot(self, a_opt, delta_opt, cost, ok):
        x, y, th, v, w  = self.car.get_state()
        self.hist_x.append(x)
        self.hist_y.append(y)

        self.path_line.set_data(self.hist_x, self.hist_y)
        self.update_car(x, y, th)

        # Draw heading arrow
        hlen = max(abs(v) * 0.4, 1.5)
        self.heading_line.set_data(
            [x, x + hlen * np.cos(th)],
            [y, y + hlen * np.sin(th)],
        )

        # Draw MPC horizon
        if self.show_horizon.get() and self.horizon_pts:
            hxs = [p[0] for p in self.horizon_pts]
            hys = [p[1] for p in self.horizon_pts]
            self.horizon_line.set_data(hxs, hys)
            self.horizon_scatter.set_offsets(np.column_stack([hxs, hys]))
        else:
            self.horizon_line.set_data([], [])
            self.horizon_scatter.set_offsets(np.empty((0, 2)))

        # Draw MPC target points
        if self.show_targets.get() and self.target_pts:
            txs = [p[0] for p in self.target_pts]
            tys = [p[1] for p in self.target_pts]
            self.target_scatter.set_offsets(np.column_stack([txs, tys]))
        else:
            self.target_scatter.set_offsets(np.empty((0, 2)))

        idx = self.frame_num % len(self.vprof)
        self.telemetry_var.set(
            f"v={abs(v):5.1f} m/s   "
            f"a={a_opt:+.2f}   "
            f"δ={np.degrees(delta_opt):+.1f}°   "
            f"cost={cost:.0f}"
        )
        self.status_var.set(
            f"Frame {self.frame_num:05d}\n"
            f"x={x:.1f}  y={y:.1f}  θ={np.degrees(th):.1f}°\n"
            f"v={abs(v):.2f} m/s  (ref {self.vprof[idx]:.2f})\n"
            f"a={a_opt:+.3f}  δ={np.degrees(delta_opt):+.2f}°\n"
            f"cost={cost:.1f}  ok={ok}"
        )
        self.canvas.draw_idle()

    # ── single MPC step ───────────────────────────────────────────────────────

    def _step(self):
        if not self.running or self.car is None or self.track is None:
            return
        
        # ─── NEW: LAP DETECTION & PATH CLEARING ──────────────────────
        # 1. Get current car position
        x, y, _, _, _ = self.car.get_state()
        
        # 2. Find the closest point on the track
        dists = np.sum((self.track - np.array([x, y])) ** 2, axis=1)
        curr_idx = int(np.argmin(dists))

        # 3. Check if we crossed the start/finish line
        # Logic: If the index just jumped from the end (high) back to the start (low)
        if hasattr(self, 'last_idx'):
            if curr_idx < self.last_idx and self.last_idx > len(self.track) * 0.8:
                self.hist_x.clear()
                self.hist_y.clear()
                self.status_var.set("🏁 Lap Completed! Path Cleared.")
        
        self.last_idx = curr_idx
        # ─────────────────────────────────────────────────────────────
        
        N          = self.p["N_horizon"].get()
        dt         = self.p["dt"].get()
        max_a      = self.p["max_accel"].get()
        max_d      = -self.p["max_decel"].get()
        max_s      = np.radians(self.p["max_steer"].get())

        weights = (
            self.w["Q_dist"].get(),
            self.w["Q_psi"].get(),
            self.w["Q_v"].get(),
            self.w["R_accel"].get(),
            self.w["R_steer"].get(),
            self.w["Rd_accel"].get(),
            self.w["Rd_steer"].get(),
        )

        # Resize u_guess if horizon changed
        if len(self.u_guess) != 2 * N:
            self.u_guess = np.zeros(2 * N)

        bounds = [(max_d, max_a), (-max_s, max_s)] * N

        # Refresh profile in case max_speed slider moved
        self.vprof = speed_profile(self.track, self.p["max_speed"].get(), self.p["min_speed"].get())

        # Propagate noise/wheelbase changes live
        self.car.noise_a          = self.p["noise_a"].get()
        self.car.noise_steer_deg  = self.p["noise_steer"].get()
        self.car.L                = self.p["wheelbase"].get()
        # Live car-model propagation (no reset required)
        self.car.Lf               = self.c["Lf"].get()
        self.car.Lr               = self.c["Lr"].get()
        self.car.m                = self.c["m"].get()
        self.car.Iz               = self.c["Iz"].get()
        self.car.Ca_f             = self.c["Ca_f"].get()
        self.car.Ca_r             = self.c["Ca_r"].get()
        self.car.mu               = self.c["mu"].get()

        Iz_val  = self.c["Iz"].get()
        Ca_f_val = self.c["Ca_f"].get()
        Ca_r_val = self.c["Ca_r"].get()
        min_lookahead = self.p["min_lookahead"].get()

        res = minimize(
            mpc_cost,
            self.u_guess,
            args=(self.car.get_state(), self.car.L, self.car.Lf, self.car.Lr,
                  self.track, N, dt, self.last_u, self.vprof, weights,
                  self.car.max_accel, self.car.max_decel, self.car.max_steer, self.car.max_speed,
                  Iz_val, Ca_f_val, Ca_r_val, min_lookahead),
            method="SLSQP",
            bounds=bounds,
            options={"ftol": 1e-3, "eps": 1e-2, "maxiter": 60},
        )

        a_opt, delta_opt = res.x.reshape(-1, 2)[0]
        pre_state = self.car.get_state()   # save BEFORE update for target computation
        self.car.update(a_opt, delta_opt, dt)

        self.u_guess      = np.roll(res.x, -2)
        self.u_guess[-2:] = res.x[-2:]
        self.last_u       = np.array([a_opt, delta_opt])

        # ── Compute predicted horizon path ────────────────────────────────────
        if self.show_horizon.get():
            hx, hy, hth, hv, hw = self.car.get_state()
            horizon_x = [hx]
            horizon_y = [hy]
            controls_opt = res.x.reshape(-1, 2)
            hbeta = np.arctan2(self.car.Lr * hw, hv) if hv > 0.1 else 0.0
            for ai, di in controls_opt:
                ai  = np.clip(ai, -self.car.max_decel, self.car.max_accel)
                di  = np.clip(di, -self.car.max_steer, self.car.max_steer)
                hv  = np.clip(hv + ai * dt, 0.1, self.car.max_speed * 1.1)
                af  = di - hbeta
                ar  = -np.arctan2(self.car.Lr * hw, hv)
                Ff  = -Ca_f_val * af
                Fr  = -Ca_r_val * ar
                hw  = hw + (self.car.Lf * Ff * np.cos(di) - self.car.Lr * Fr) / Iz_val * dt
                hth = hth + hw * dt
                hx  = hx + hv * np.cos(hth) * dt
                hy  = hy + hv * np.sin(hth) * dt
                hbeta = np.arctan2(self.car.Lr * hw, hv) if hv > 0.1 else 0.0
                horizon_x.append(hx)
                horizon_y.append(hy)
            self.horizon_pts = list(zip(horizon_x, horizon_y))
        else:
            self.horizon_pts = []

        # ── Compute MPC target points (one per horizon step) ──────────────────
        if self.show_targets.get():
            px0, py0, _, pv0, _ = pre_state
            dists0 = np.sum((self.track - np.array([px0, py0])) ** 2, axis=1)
            tidx   = int(np.argmin(dists0))
            tp1    = self.track[tidx]
            tp2    = self.track[(tidx + 1) % len(self.track)]
            t_dpi  = max(np.linalg.norm(tp2 - tp1), 0.1)
            t_mla  = min_lookahead
            tv     = pv0
            target_xs, target_ys = [], []
            controls_opt = res.x.reshape(-1, 2)
            for i, (tai, _tdi) in enumerate(controls_opt):
                tai = np.clip(tai, -self.car.max_decel, self.car.max_accel)
                tv  = np.clip(tv + tai * dt, 0.1, self.car.max_speed * 1.1)
                spd_dist  = tv * (dt * (i + 1))
                pred_dist = max(t_mla, spd_dist)
                t_off = int(round(pred_dist / t_dpi))
                ri    = (tidx + t_off) % len(self.track)
                target_xs.append(self.track[ri, 0])
                target_ys.append(self.track[ri, 1])
            self.target_pts = list(zip(target_xs, target_ys))
        else:
            self.target_pts = []

        self._update_plot(a_opt, delta_opt, res.fun, res.success)
        self.frame_num += 1

        ms = max(1, int(dt * 100))
        self._after_id = self.root.after(ms, self._step)

    # ── playback control ──────────────────────────────────────────────────────

    def _do_play(self):
        if self.running:
            return
        self.running = True
        self.btn_play.configure(bg="#0a3a1a")
        self.btn_pause.configure(bg="#4a3a1a")
        self._step()

    def _do_pause(self):
        self.running = False
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.btn_play.configure(bg="#1a4a2a")
        self.btn_pause.configure(bg="#2a1a0a")
        self.status_var.set(self.status_var.get() + "\n[PAUSED]")

    def _do_reset(self):
        self._do_pause()
        self._reset_state()
        self._redraw_track()
        self.telemetry_var.set("")
        self.status_var.set("Reset — press ▶ Play to start")

    # ── track actions ─────────────────────────────────────────────────────────

    def _do_gen_track(self):
        was = self.running
        self._do_pause()
        self.track = generate_track(
            step_distance=self.p["step_dist"].get(),
            n_control=self.p["n_control"].get(),
            seed=int(self.p["track_seed"].get()),
        )
        self._reset_state()
        self._redraw_track()
        self.status_var.set("New track generated ✓")
        if was:
            self._do_play()

    def _do_save_track(self):
        if self.track is None:
            messagebox.showwarning("No Track", "Generate a track first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Track",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("NumPy", "*.npy")],
        )
        if not path:
            return
        try:
            if path.endswith(".npy"):
                np.save(path, self.track)
            else:
                np.savetxt(path, self.track, delimiter=",",
                           header="x,y", comments="")
            self.status_var.set(f"Saved → {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _do_load_midpoints(self):
        path = filedialog.askopenfilename(
            title="Load Midpoints",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if not path:
            return

        was = self.running
        self._do_pause()

        try:
            # Load the raw points
            points = np.loadtxt(path, delimiter=",", skiprows=1)

            # Validate we have at least 3 points
            if len(points) < 3:
                raise ValueError("Need at least 3 points to form a track")

            # Reorder points to create a smooth path
            def order_points(points):
                # First try to find a continuous path
                tree = KDTree(points)
                start_idx = np.argmin(np.linalg.norm(points, axis=1))  # Start near origin

                ordered = [start_idx]
                unvisited = set(range(len(points)))
                unvisited.remove(start_idx)

                while unvisited:
                    last_point = points[ordered[-1]]
                    # Find nearest unvisited point
                    dist, idx = tree.query(last_point, k=len(points))
                    for i in idx:
                        if i in unvisited:
                            ordered.append(i)
                            unvisited.remove(i)
                            break

                return points[ordered]

            # Apply ordering
            ordered_points = order_points(points)

            # Store the ordered points
            self.track = ordered_points
            self._reset_state()
            self._redraw_track()
            self.status_var.set(f"Loaded midpoints ← {path}")

        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return
        finally:
            if was:
                self._do_play()
            
    
    def _do_load_track(self):
        path = filedialog.askopenfilename(
            title="Load Track",
            filetypes=[("CSV", "*.csv"), ("NumPy", "*.npy"), ("All", "*.*")],
        )
        if not path:
            return
        was = self.running
        self._do_pause()
        try:
            if path.endswith(".npy"):
                self.track = np.load(path)
            else:
                self.track = np.loadtxt(path, delimiter=",", skiprows=1)
            self._reset_state()
            self._redraw_track()
            self.status_var.set(f"Loaded ← {path}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return
        if was:
            self._do_play()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x780")
    root.resizable(True, True)
    app = MPCCarApp(root)
    root.mainloop()
