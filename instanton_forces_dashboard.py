import threading
import time
from collections import deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ---------------------- Physical constants (atomic units) ----------------------
# Atomic unit of energy (Hartree) and Boltzmann in Hartree/K
try:
    from scipy import constants
    Eh = constants.physical_constants['atomic unit of energy'][0]
    kB = constants.Boltzmann / Eh  # Hartree / K
    eV_to_Eh = constants.eV / Eh
except Exception:
    # Fallback values (should be very close)
    Eh = 4.3597447222071e-18
    kB = 3.166811563e-6
    eV_to_Eh = 0.03674932217565499

# ---------------------- Force field (Asymmetric Eckart) -----------------------
def V_Asym_Eckart(q, V0, a, alpha):
    # V(q) = V0 * (1-alpha)/(1+exp(-2 a q)) + V0 * (1+sqrt(alpha))^2 / (4 cosh^2(a q))
    return V0 * (1.0 - alpha) / (1.0 + np.exp(-2.0 * a * q)) + V0 * (1.0 + np.sqrt(alpha))**2 / (4.0 * np.cosh(a*q)**2)

def F_Asym_Eckart(q, V0, a, alpha):
    # F(q) = -dV/dq
    term1 = V0 * a * (1.0 + np.sqrt(alpha))**2 * np.sinh(a*q) / (2.0 * np.cosh(a*q)**3)
    term2 = 2.0 * V0 * a * (1.0 - alpha) * np.exp(-2.0*a*q) / (1.0 + np.exp(-2.0*a*q))**2
    return term1 - term2

# ------------------- Ring-polymer utilities (pure NumPy) ----------------------
def normal_mode_frequencies(beta_n, n):
    omega_n = 1.0 / beta_n
    k = np.arange(n)
    return 2.0 * omega_n * np.sin(np.pi * k / n)  # omega[0] = 0

def mode_propagation(q, v, omega, delta_t):
    # FFT to normal modes, evolve harmonically, IFFT back
    Qk = np.fft.fft(q)
    Vk = np.fft.fft(v)

    Qk_next = np.empty_like(Qk, dtype=np.complex128)
    Vk_next = np.empty_like(Vk, dtype=np.complex128)

    # Zero-frequency mode: free particle
    Qk_next[0] = Qk[0] + Vk[0] * delta_t
    Vk_next[0] = Vk[0]

    omega_pos = omega[1:]
    coswt = np.cos(omega_pos * delta_t)
    sinwt = np.sin(omega_pos * delta_t)
    Qk_pos = Qk[1:]
    Vk_pos = Vk[1:]
    Qk_next[1:] = Qk_pos * coswt + Vk_pos * (sinwt / omega_pos)
    Vk_next[1:] = Vk_pos * coswt - Qk_pos * (omega_pos * sinwt)

    q_next = np.fft.ifft(Qk_next).real
    v_next = np.fft.ifft(Vk_next).real
    return q_next, v_next

def polymer_step_rattle(q, v, beta_n, delta_t, m, force_func, ff_params, q0, qP, P):
    """
    One time step with:
      1) v half-kick by external force
      2) exact free-ring-polymer propagation in normal modes
      3) re-enforce constraints on positions and velocities (0 and P)
      4) another v half-kick by external force
    """
    n = q.size
    omega = normal_mode_frequencies(beta_n, n)

    # Half-kick
    f = force_func(q, *ff_params)
    v_half = v + 0.5 * delta_t * f / m

    # Free-RP propagate (exact)
    q_mode, v_mode = mode_propagation(q, v_half, omega, delta_t)

    # Enforce constraints
    q_mode[0] = q0
    q_mode[P] = qP

    # Another half-kick with updated positions
    f_new = force_func(q_mode, *ff_params)
    v_new = v_mode + 0.5 * delta_t * f_new / m
    v_new[0] = 0.0
    v_new[P] = 0.0
    return q_mode, v_new

def virial_weights(P, n):
    k = np.arange(n)
    s = np.zeros(n)
    # s weights bead 0 (x1) and t weights bead P (x2)
    s[:P+1] = 1.0 - (k[:P+1] / P)
    s[P:]   = (k[P:] - P) / P
    t = 1.0 - s
    s[0], s[P] = 1.0, 0.0
    t[0], t[P] = 0.0, 1.0
    return s, t

def dU_dq_all(q, k_eff, force_func, ff_params):
    springs = k_eff * (2.0*q - np.roll(q,1) - np.roll(q,-1))
    pot     = - force_func(q, *ff_params)  # because F = -dV/dq
    return springs + pot

# -------------------------- Sampler (worker thread) ---------------------------
class Sampler:
    def __init__(self):
        # Default parameters
        self.T = 1000.0               # K
        self.P = 16                  # half-beads
        self.x_bar = 0.0
        self.delta_x = 0.0
        self.m = 1060.0              # atomic mass units mapped as given (AU of mass, consistent with user's units)
        self.V0_eV = 0.425           # barrier height in eV
        self.a = 1.36
        self.alpha = 1.25

        # Integrator settings
        self.t_end = 100.0
        self.delta_t = 2.0
        self.samples_per_frame = 40   # do this many samples between UI updates

        # State
        self._lock = threading.Lock()
        self._paused = False
        self._stop = False
        self._needs_reset = True

        # Running stats
        self.reset_stats()
        # latest configuration for plotting
        self.latest_q = None

    def current_params(self):
        with self._lock:
            return dict(T=self.T, P=int(self.P), x_bar=self.x_bar, delta_x=self.delta_x,
                        m=self.m, V0_eV=self.V0_eV, a=self.a, alpha=self.alpha,
                        t_end=self.t_end, delta_t=self.delta_t)

    def apply_params(self, T, P, x_bar, delta_x, m, V0_eV, a, alpha):
        with self._lock:
            self.T = float(T)
            self.P = int(P)
            self.x_bar = float(x_bar)
            self.delta_x = float(delta_x)
            self.m = float(m)
            self.V0_eV = float(V0_eV)
            self.a = float(a)
            self.alpha = float(alpha)
            self._needs_reset = True  # trigger rebuild of polymer & weights

    def toggle_pause(self):
        with self._lock:
            self._paused = not self._paused
            return not self._paused

    def request_reset(self):
        with self._lock:
            self._needs_reset = True

    def stop(self):
        with self._lock:
            self._stop = True

    # --------- statistics handling ---------
    def reset_stats(self):
        self.count = 0
        self.mean_bar_TD = 0.0
        self.mean_bar_VI = 0.0
        self.mean_del_TD = 0.0
        self.mean_del_VI = 0.0
        self.history_bar_TD = deque(maxlen=5000)
        self.history_bar_VI = deque(maxlen=5000)
        self.history_del_TD = deque(maxlen=5000)
        self.history_del_VI = deque(maxlen=5000)

    def _update_means(self, bar_TD, bar_VI, del_TD, del_VI):
        self.count += 1
        c = self.count
        self.mean_bar_TD += (bar_TD - self.mean_bar_TD) / c
        self.mean_bar_VI += (bar_VI - self.mean_bar_VI) / c
        self.mean_del_TD += (del_TD - self.mean_del_TD) / c
        self.mean_del_VI += (del_VI - self.mean_del_VI) / c
        self.history_bar_TD.append(self.mean_bar_TD)
        self.history_bar_VI.append(self.mean_bar_VI)
        self.history_del_TD.append(self.mean_del_TD)
        self.history_del_VI.append(self.mean_del_VI)

    # --------- main worker loop ---------
    def run(self):
        rng = np.random.default_rng()
        while True:
            with self._lock:
                if self._stop:
                    return
                paused = self._paused
                needs_reset = self._needs_reset
                T = self.T; P = int(self.P); x_bar = self.x_bar; delta_x = self.delta_x
                m = self.m; V0 = self.V0_eV * eV_to_Eh; a = self.a; alpha = self.alpha
                t_end = self.t_end; delta_t = self.delta_t

            if paused:
                time.sleep(0.05)
                continue

            if needs_reset:
                # Build polymer, weights, and cached quantities
                n = 2 * P
                beta = 1.0 / (kB * T)
                beta_n = beta / n
                k_eff = m / (beta_n**2)
                s, t = virial_weights(P, n)

                q0 = x_bar - delta_x
                qP = x_bar + delta_x

                # initial polymer: cosine arc between endpoints
                q = x_bar - np.cos(2.0*np.pi * np.arange(n)/n) * delta_x
                v = rng.normal(0.0, 1.0/np.sqrt(beta_n*m), size=n)
                # Reset stats
                self.reset_stats()
                with self._lock:
                    self._needs_reset = False

            # Do several samples per UI frame for smoothness
            # ensure latest_q available
            if self.latest_q is None:
                self.latest_q = q.copy()
            for _ in range(self.samples_per_frame):
                # draw velocities from Boltzmann at each "trajectory"
                v = rng.normal(0.0, 1.0/np.sqrt(beta_n*m), size=n)

                # evolve for N_time = int(t_end/delta_t)
                N_time = max(1, int(t_end/delta_t))
                for _step in range(N_time):
                    q, v = polymer_step_rattle(
                        q, v, beta_n, delta_t, m,
                        F_Asym_Eckart, (V0, a, alpha), q0, qP, P
                    )

                # TD estimators at endpoints
                F0 = F_Asym_Eckart(q0, V0, a, alpha)
                FP = F_Asym_Eckart(qP, V0, a, alpha)
                force_0_TD = -F0 + k_eff*(2.0*q0 - q[-1] - q[1])
                force_P_TD = -FP + k_eff*(2.0*qP - q[P-1] - q[P+1])

                # VI estimators
                dUdq = dU_dq_all(q, k_eff, F_Asym_Eckart, (V0, a, alpha))
                force_0_VI = np.dot(s, dUdq)
                force_P_VI = np.dot(t, dUdq)

                n = 2*P  # for clarity
                bar_TD = (force_0_TD + force_P_TD) / n
                del_TD = (-force_0_TD + force_P_TD) / n
                bar_VI = (force_0_VI + force_P_VI) / n
                del_VI = (-force_0_VI + force_P_VI) / n

                self._update_means(bar_TD, bar_VI, del_TD, del_VI)

                # store latest configuration for plot 3
                self.latest_q = q.copy()

            # Allow UI thread to catch up
            time.sleep(0.01)

# ------------------------------- Plotting/UI ----------------------------------
def main():
    sampler = Sampler()

    # Two figures: one for x̄ direction; one for Δx direction (tooling prefers no subplots)
    fig1, ax1 = plt.subplots(figsize=(7.0, 4.0))
    line_bar_TD, = ax1.plot([], [], lw=1.5, label=r"$\bar{x}$ TD")
    line_bar_VI, = ax1.plot([], [], lw=1.5, label=r"$\bar{x}$ VI")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Running mean force (a.u.)")
    ax1.set_title(r"Mean polymer force along $\bar{x}$")
    ax1.legend(loc="best")
    ax1.grid(True)

    fig2, ax2 = plt.subplots(figsize=(7.0, 4.0))
    ax2.grid(True)
    line_del_TD, = ax2.plot([], [], lw=1.5, label=r"$\delta x$ TD")
    line_del_VI, = ax2.plot([], [], lw=1.5, label=r"$\delta x$ VI")
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Running mean force (a.u.)")
    ax2.set_title(r"Mean polymer force along $\delta x$")
    ax2.legend(loc="best")
    
    # Figure 3: Potential + bead ring (polymer configuration view)
    fig3, ax3 = plt.subplots(figsize=(7.0, 4.0))
    line_V, = ax3.plot([], [], lw=1.5, label="V(x)")
    ring_line, = ax3.plot([], [], "--", lw=1.5, label="bead ring")
    ring_scatter = ax3.plot([], [], "o", ms=4)[0]
    center_marker = ax3.plot([], [], "o", mfc="white", mec="black", ms=6)[0]
    ax3.set_xlabel(r"$x$"); ax3.set_ylabel(r"$V(x)$"); ax3.set_title("Potential with bead ring")
    ax3.legend(loc="best"); ax3.grid(True)

    # New Figure 4: controls panel (empty canvas with sliders & buttons)
    fig4, ax4 = plt.subplots(figsize=(10.0, 6.0))
    ax4.axis('off')
    axcolor = 'lightgoldenrodyellow'

    # Controls moved to fig4
    ax_T      = fig4.add_axes([0.10, 0.80, 0.35, 0.05], facecolor=axcolor)
    ax_P      = fig4.add_axes([0.10, 0.72, 0.35, 0.05], facecolor=axcolor)
    ax_xbar   = fig4.add_axes([0.10, 0.64, 0.35, 0.05], facecolor=axcolor)
    ax_delx   = fig4.add_axes([0.10, 0.56, 0.35, 0.05], facecolor=axcolor)
    ax_V0     = fig4.add_axes([0.55, 0.80, 0.35, 0.05], facecolor=axcolor)
    ax_a      = fig4.add_axes([0.55, 0.72, 0.35, 0.05], facecolor=axcolor)
    ax_alpha  = fig4.add_axes([0.55, 0.64, 0.35, 0.05], facecolor=axcolor)
    ax_m      = fig4.add_axes([0.55, 0.56, 0.35, 0.05], facecolor=axcolor)

    s_T     = Slider(ax_T,     r"$T$ / $\mathrm{K}$",       10.0, 5000.0,  valinit=sampler.T, valstep=1.0)
    s_P     = Slider(ax_P,     r"$P$",           2,    256,     valinit=sampler.P, valstep=1.0)
    s_xbar  = Slider(ax_xbar,  r"$\bar{x}$",          -6.0,  6.0,     valinit=sampler.x_bar, valstep=0.01)
    s_delx  = Slider(ax_delx,  r"$\delta x$",          0.0,   6.0,    valinit=sampler.delta_x, valstep=0.01)
    s_V0    = Slider(ax_V0,    r"$V_0$ / $\mathrm{eV}$",     0.01,  2.0,    valinit=sampler.V0_eV, valstep=0.001)
    s_a     = Slider(ax_a,     r"$a$",           0.05,  5.0,    valinit=sampler.a, valstep=0.01)
    s_alpha = Slider(ax_alpha, r"$\alpha$",       0.05,  5.0,    valinit=sampler.alpha, valstep=0.01)
    s_m     = Slider(ax_m,     r"$m$",           1.0,   5000.0, valinit=sampler.m, valstep=1.0)

    # Buttons
    ax_apply = fig4.add_axes([0.10, 0.18, 0.18, 0.06])
    ax_pause = fig4.add_axes([0.32, 0.18, 0.18, 0.06])
    ax_reset = fig4.add_axes([0.54, 0.18, 0.18, 0.06])

    b_apply = Button(ax_apply, "Apply Params")
    b_pause = Button(ax_pause, "Pause/Resume")
    b_reset = Button(ax_reset, "Reset")

    # Button callbacks
    def on_apply(event):
        sampler.apply_params(
            s_T.val, int(s_P.val), s_xbar.val, s_delx.val, s_m.val, s_V0.val, s_a.val, s_alpha.val
        )

    def on_pause(event):
        running = sampler.toggle_pause()
        # Visual feedback in the button label
        b_pause.label.set_text("Pause" if running else "Resume")

    def on_reset(event):
        sampler.request_reset()

    b_apply.on_clicked(on_apply)
    b_pause.on_clicked(on_pause)
    b_reset.on_clicked(on_reset)

    # Start worker thread
    worker = threading.Thread(target=sampler.run, daemon=True)
    worker.start()

    # UI update loop (timer-based)
    def update_lines(_evt):
        # --- Figure 3: potential + bead ring ---
        xmin, xmax = -4.0, 4.0
        xs = np.linspace(xmin, xmax, 800)
        params = sampler.current_params()
        Vx = V_Asym_Eckart(xs, params["V0_eV"]*eV_to_Eh, params["a"], params["alpha"])
        line_V.set_data(xs, Vx)
        ax3.set_xlim(xmin, xmax)
        # default y-lims based on potential
        y0 = float(np.max(Vx)) * 0.7
        amp = 0.20 * (np.max(Vx) - np.min(Vx) + 1e-9)
        y_min = min(np.min(Vx), y0 - 2.2*amp)
        y_max = max(np.max(Vx), y0 + 2.2*amp)
        ax3.set_ylim(y_min, y_max)
        q = sampler.latest_q
        if q is not None:
            n = q.size
            k = np.arange(n)
            xk = q.astype(float)
            yk = y0 + amp * np.cos(2.0*np.pi*k/n)
            ring_line.set_data(np.r_[xk, xk[0]], np.r_[yk, yk[0]])
            ring_scatter.set_data(xk, yk)
            ax3.set_title(f"Potential with bead ring (n={n})")
        else:
            ring_line.set_data([], [])
            ring_scatter.set_data([], [])
        center_marker.set_data([params["x_bar"]], [y0])

        # pull histories and update plots
        y1 = np.fromiter(sampler.history_bar_TD, dtype=float, count=len(sampler.history_bar_TD))
        y2 = np.fromiter(sampler.history_bar_VI, dtype=float, count=len(sampler.history_bar_VI))
        x1 = np.arange(1, len(y1)+1)
        x2 = np.arange(1, len(y2)+1)
        line_bar_TD.set_data(x1, y1)
        line_bar_VI.set_data(x2, y2)
        if len(x1) > 5:
            ax1.set_xlim(1, len(x1))
            # autoscale y with margin
            ymin = np.nanmin([y1.min() if len(y1) else 0.0, y2.min() if len(y2) else 0.0])
            ymax = np.nanmax([y1.max() if len(y1) else 0.0, y2.max() if len(y2) else 0.0])
            if ymin == ymax:
                ymin -= 1e-6; ymax += 1e-6
            ax1.set_ylim(ymin - 0.05*abs(ymin), ymax + 0.05*abs(ymax))

        y3 = np.fromiter(sampler.history_del_TD, dtype=float, count=len(sampler.history_del_TD))
        y4 = np.fromiter(sampler.history_del_VI, dtype=float, count=len(sampler.history_del_VI))
        x3 = np.arange(1, len(y3)+1)
        x4 = np.arange(1, len(y4)+1)
        line_del_TD.set_data(x3, y3)
        line_del_VI.set_data(x4, y4)
        if len(x3) > 5:
            ax2.set_xlim(1, len(x3))
            ymin = np.nanmin([y3.min() if len(y3) else 0.0, y4.min() if len(y4) else 0.0])
            ymax = np.nanmax([y3.max() if len(y3) else 0.0, y4.max() if len(y4) else 0.0])
            if ymin == ymax:
                ymin -= 1e-6; ymax += 1e-6
            ax2.set_ylim(ymin - 0.05*abs(ymin), ymax + 0.05*abs(ymax))

        fig1.canvas.draw_idle()
        fig2.canvas.draw_idle()
        fig3.canvas.draw_idle()

    # Timer to refresh plots ~10 times/sec
    timer1 = fig1.canvas.new_timer(interval=100)
    timer1.add_callback(update_lines, None)
    timer1.start()

    timer2 = fig2.canvas.new_timer(interval=100)
    timer2.add_callback(update_lines, None)
    timer2.start()
    timer3 = fig3.canvas.new_timer(interval=150)
    timer3.add_callback(update_lines, None)
    timer3.start()

    # Graceful close
    def on_close(_event):
        sampler.stop()

    fig1.canvas.mpl_connect('close_event', on_close)
    fig2.canvas.mpl_connect('close_event', on_close)
    fig3.canvas.mpl_connect('close_event', on_close)
    fig4.canvas.mpl_connect('close_event', on_close)

    plt.show()

if __name__ == "__main__":
    main()