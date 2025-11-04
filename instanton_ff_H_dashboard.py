import threading, time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ---------------------- constants (a.u.) ----------------------
try:
    from scipy import constants
    Eh = constants.physical_constants['atomic unit of energy'][0]
    kB = constants.Boltzmann / Eh           # Hartree / K
    eV_to_Eh = constants.eV / Eh            # Hartree per eV
except Exception:
    kB = 3.166811563e-6
    eV_to_Eh = 0.03674932217565499

# ---------------------- potential & forces --------------------
def V_Asym_Eckart(q, V0, a, alpha):
    return V0 * (1.0 - alpha) / (1.0 + np.exp(-2.0 * a * q)) + V0 * (1.0 + np.sqrt(alpha))**2 / (4.0 * np.cosh(a*q)**2)

def F_Asym_Eckart(q, V0, a, alpha):
    term1 = V0 * a * (1.0 + np.sqrt(alpha))**2 * np.sinh(a*q) / (2.0 * np.cosh(a*q)**3)
    term2 = 2.0 * V0 * a * (1.0 - alpha) * np.exp(-2.0*a*q) / (1.0 + np.exp(-2.0*a*q))**2
    return term1 - term2

def ddV_central(q, V0, a, alpha, h=1e-4):
    return (V_Asym_Eckart(q+h, V0, a, alpha) - 2.0*V_Asym_Eckart(q, V0, a, alpha) + V_Asym_Eckart(q-h, V0, a, alpha)) / (h*h)

# ---------------------- ring polymer stepping -----------------
def normal_mode_frequencies(beta_n, n):
    k = np.arange(n)
    return 2.0 / beta_n * np.sin(np.pi * k / n)

def mode_propagation(q, v, omega, dt):
    Qk, Vk = np.fft.fft(q), np.fft.fft(v)
    Qk_next = np.empty_like(Qk, dtype=np.complex128)
    Vk_next = np.empty_like(Vk, dtype=np.complex128)
    Qk_next[0] = Qk[0] + Vk[0] * dt
    Vk_next[0] = Vk[0]
    om = omega[1:]
    c, s = np.cos(om*dt), np.sin(om*dt)
    Qk_pos, Vk_pos = Qk[1:], Vk[1:]
    Qk_next[1:] = Qk_pos * c + Vk_pos * (s/om)
    Vk_next[1:] = Vk_pos * c - Qk_pos * (om*s)
    return np.fft.ifft(Qk_next).real, np.fft.ifft(Vk_next).real

def polymer_step_rattle(q, v, beta_n, dt, m, force_func, ff_params, q0, qP, P):
    omega = normal_mode_frequencies(beta_n, q.size)
    f = force_func(q, *ff_params)
    v_half = v + 0.5 * dt * f / m
    q_mode, v_mode = mode_propagation(q, v_half, omega, dt)
    q_mode[0], q_mode[P] = q0, qP
    f_new = force_func(q_mode, *ff_params)
    v_new = v_mode + 0.5 * dt * f_new / m
    v_new[0] = 0.0; v_new[P] = 0.0
    return q_mode, v_new

# ---------------------- estimators helpers --------------------
def build_qstar(q0, qP, P, n):
    k = np.arange(n)
    q_star = np.empty(n)
    q_star[:P+1] = q0 + (qP - q0) * (k[:P+1] / P)
    q_star[P:]   = qP + (q0 - qP) * ((k[P:] - P) / P)
    return q_star

def signs_TD(P, n):
    F_first_sign = -np.ones(n); F_first_sign[1:P+1] = 1.0
    F_sec_sign = np.ones(n); F_sec_sign[P:] = -1.0; F_sec_sign[0] = 0.0; F_sec_sign[P] = 0.0
    return F_first_sign, F_sec_sign

def signs_VI(P, n):
    F_sign = -np.ones(n); F_sign[1:P+1] = 1.0; F_sign[0] = 0.0
    return F_sign

# ---------------------- online stats (Welford) ----------------
class OnlineStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def add(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    @property
    def var(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0
    @property
    def sem(self):
        return np.sqrt(self.var / self.n) if self.n > 1 else 0.0

# ---------------------- ACF/SEM helpers -----------------------
def acf_fft(x):
    """Return normalized autocorrelation function rho[k] for k=0..n-1 using FFT."""
    x = np.asarray(x)
    x = x - np.mean(x)
    n = x.size
    if n < 2:
        return np.array([1.0])
    nfft = 1 << (2*n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[:n]
    ac /= ac[0] if ac[0] != 0 else 1.0
    return ac

def tau_int_from_acf(rho, max_lag=None):
    """Sum rho until first non-positive or max_lag; return tau_int >= 0.5."""
    n = rho.size
    L = n-1 if max_lag is None else min(int(max_lag), n-1)
    s = 0.0
    for k in range(1, L+1):
        if rho[k] <= 0.0:
            break
        s += rho[k]
    return 0.5 + s

# ---------------------- worker -------------------------------
class Sampler:
    def __init__(self):
        # User-facing parameters (defaults)
        self.T = 200.0
        self.P = 80
        self.x_bar = -0.1
        self.delta_x = 0.72
        self.m = 1060.0
        self.V0_eV = 0.425
        self.a = 1.36
        self.alpha = 1.25

        # Internal evolution controls (no sliders)
        self.t_end = 150.0
        self.delta_t = 5.0
        self.N_pre = 10000

        self.samples_per_frame = 10
        self._lock = threading.Lock()
        self._paused = False
        self._stop = False
        self._needs_reset = True

        # Running histories (large so X window can exceed 5000)
        self.history_cff = deque(maxlen=200000)
        self.history_dH_TD = deque(maxlen=200000)
        self.history_dH_VI = deque(maxlen=200000)
        # Histogram/ACF buffer for C_ff/C_dd samples
        self.samples_cff = deque(maxlen=100000)  # rolling window

        # Online stats
        self.stats_cff = OnlineStats()
        self.stats_S_TD = OnlineStats()
        self.stats_S_VI = OnlineStats()

        # ACF control
        self.acf_max_lag = 2000

        # Display window (x-range) for plots 1 & 2
        self.display_window = 3000  # initial samples to show

        # latest configuration
        self.latest_q = None

    def apply_params(self, T, P, x_bar, delta_x, m, V0_eV, a, alpha):
        with self._lock:
            self.T=float(T); self.P=int(P); self.x_bar=float(x_bar); self.delta_x=float(delta_x)
            self.m=float(m); self.V0_eV=float(V0_eV); self.a=float(a); self.alpha=float(alpha)
            self._needs_reset = True

    def set_acf_maxlag(self, L):
        with self._lock:
            self.acf_max_lag = int(L)

    def set_display_window(self, W):
        with self._lock:
            self.display_window = int(W)

    def toggle_pause(self):
        with self._lock:
            self._paused = not self._paused
            return not self._paused

    def request_reset(self):
        with self._lock:
            self._needs_reset = True

    def stop(self): 
        with self._lock: self._stop = True

    def run(self):
        rng = np.random.default_rng()
        while True:
            with self._lock:
                if self._stop: return
                paused = self._paused; needs_reset = self._needs_reset
                T=self.T; P=int(self.P); x_bar=self.x_bar; delta_x=self.delta_x
                m=self.m; V0=self.V0_eV*eV_to_Eh; a=self.a; alpha=self.alpha
                t_end=self.t_end; dt=self.delta_t; N_pre=self.N_pre
            if paused: time.sleep(0.05); continue
            if needs_reset:
                n=2*P; beta=1.0/(kB*T); beta_n=beta/n
                q0=x_bar-delta_x; qP=x_bar+delta_x
                q = x_bar - np.cos(2.0*np.pi*np.arange(n)/n) * delta_x
                q_star=build_qstar(q0,qP,P,n)
                F_first_sign,F_sec_sign=signs_TD(P,n)
                F_sign=signs_VI(P,n)
                # pre-equilibrate
                v = rng.normal(0.0, 1.0/np.sqrt(beta_n*m), size=n)
                for _ in range(max(0, N_pre)):
                    q, v = polymer_step_rattle(q, v, beta_n, dt, m, F_Asym_Eckart, (V0,a,alpha), q0, qP, P)
                    if (_+1) % 50 == 0:
                        v = rng.normal(0.0, 1.0/np.sqrt(beta_n*m), size=n)
                # reset histories/stats
                self.history_cff.clear()
                self.history_dH_TD.clear()
                self.history_dH_VI.clear()
                self.samples_cff.clear()
                self.stats_cff = OnlineStats()
                self.stats_S_TD = OnlineStats()
                self.stats_S_VI = OnlineStats()
                self.latest_q = q.copy()
                with self._lock:
                    self._needs_reset=False
                    self._cached=dict(n=n,beta=beta,beta_n=beta_n,q0=q0,qP=qP,q=q,q_star=q_star,
                                      F_first_sign=F_first_sign,F_sec_sign=F_sec_sign,F_sign=F_sign,
                                      dt=dt,m=m,V0=V0,a=a,alpha=alpha,P=P,t_end=t_end)

            cache=getattr(self,"_cached",None)
            if cache is None: time.sleep(0.01); continue
            n=cache['n']; beta=cache['beta']; beta_n=cache['beta_n']
            q0=cache['q0']; qP=cache['qP']; q=cache['q']; q_star=cache['q_star']
            F_first_sign=cache['F_first_sign']; F_sec_sign=cache['F_sec_sign']; F_sign=cache['F_sign']
            dt=cache['dt']; m=cache['m']; V0=cache['V0']; a=cache['a']; alpha=cache['alpha']; P=cache['P']; t_end=cache['t_end']

            for _ in range(self.samples_per_frame):
                v = rng.normal(0.0, 1.0/np.sqrt(beta_n*m), size=n)
                N_time=max(1,int(t_end/dt))
                for _step in range(N_time):
                    q, v = polymer_step_rattle(q, v, beta_n, dt, m, F_Asym_Eckart, (V0,a,alpha), q0, qP, P)

                # f_v for C_ff/C_dd
                fv_raw = (q[1]-q[-1])*(q[P+1]-q[P-1])
                cff_sample = -m * (P/beta)**2 * fv_raw

                # common
                q_roll=np.roll(q,1); diff_sq=(q-q_roll)**2
                V_q=V_Asym_Eckart(q,V0,a,alpha); dV_q=-F_Asym_Eckart(q,V0,a,alpha); ddV_q=ddV_central(q,V0,a,alpha)

                # TD
                F_first = -m * n * np.sum(F_first_sign*diff_sq) / (beta**2)
                F_sec   = np.sum(F_sec_sign*V_q) / P
                F_TD = F_first + F_sec
                G_TD = np.sum(diff_sq)
                S_TD = F_TD**2 + 2.0*n/(beta**2) - 4.0*m*n*G_TD/(beta**3)

                # VI
                q_min_q_star = q - q_star
                F_VI = -np.sum(F_sign * (0.5*q_min_q_star*dV_q + V_q)) / P
                G_VI = np.sum(3.0*q_min_q_star*dV_q + q_min_q_star**2 * ddV_q)
                S_VI = F_VI**2 + 4.0/(beta**2) - 16.0*m*(q0-qP)**2/(beta**3) - G_VI/(n*beta)

                # online stats & buffers
                self.stats_cff.add(cff_sample)
                self.samples_cff.append(cff_sample)
                self.stats_S_TD.add(S_TD)
                self.stats_S_VI.add(S_VI)

                # displayed means
                self.history_cff.append(self.stats_cff.mean)
                mean_S_TD = self.stats_S_TD.mean; mean_S_VI = self.stats_S_VI.mean
                self.history_dH_TD.append(np.sqrt(max(0.0, 0.5*mean_S_TD)))
                self.history_dH_VI.append(np.sqrt(max(0.0, 0.5*mean_S_VI)))

                self.latest_q = q.copy()

            cache['q']=q
            time.sleep(0.01)

# ---------------------- UI -------------------------------
def main():
    s = Sampler()

    # Figure 1: C_ff/C_dd with corr-aware SEM shading (computed in refresh using ACF window)
    fig1, ax1 = plt.subplots(figsize=(7,4))
    line_cff, = ax1.plot([], [], lw=1.5)
    ax1.set_xlabel("Samples"); ax1.set_ylabel("Running mean"); ax1.set_title(r"$C_{ff}(0)/C_{dd}(0)$"); ax1.grid(True)
    txt1 = ax1.text(0.02, 0.98, "", transform=ax1.transAxes, va='top', ha='left')
    sem_band = [None]

    # Figure 2: ΔH
    fig2, ax2 = plt.subplots(figsize=(7,4))
    line_dH_TD, = ax2.plot([], [], lw=1.5, label="Thermodynamic")
    line_dH_VI, = ax2.plot([], [], lw=1.5, label="Virial")
    ax2.set_xlabel("Samples"); ax2.set_ylabel("Running mean"); ax2.set_title(r"$\Delta H$ estimators"); ax2.legend(); ax2.grid(True)
    txt2 = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, va='top', ha='left')

    # Figure 3: Potential with bead ring (controls live under this one)
    fig3, ax3 = plt.subplots(figsize=(7,4))
    line_V, = ax3.plot([], [], lw=1.5, label="V(x)")
    ring_line, = ax3.plot([], [], "--", lw=1.5, label="bead ring")
    ring_scatter = ax3.plot([], [], "o", ms=4)[0]
    center_marker = ax3.plot([], [], "o", mfc="white", mec="black", ms=6)[0]
    ax3.set_xlabel(r"$x$"); ax3.set_ylabel(r"$V(x)$"); ax3.set_title("Potential with bead ring"); ax3.legend(); ax3.grid(True)

    # Figure 4: Histogram of C_ff/C_dd samples
    fig4, ax4 = plt.subplots(figsize=(7,4))
    ax4.set_title(r"Histogram of $f_v(\mathbf{q})$")
    ax4.set_xlabel(r"$C_{ff}(0)/C_{dd}(0)$"); ax4.set_ylabel("Count")

    # Leave space under fig3 for widgets
    fig1.subplots_adjust(bottom=0.12)
    fig2.subplots_adjust(bottom=0.12)
    fig3.subplots_adjust(bottom=0.12)
    # New Figure 5: controls panel
    fig5, ax5 = plt.subplots(figsize=(10,6))
    ax5.axis('off')
    fig4.subplots_adjust(bottom=0.12)

    # Sliders on fig3
    axcolor='lightgoldenrodyellow'
    ax_T     = fig5.add_axes([0.10,0.80,0.35,0.06], facecolor=axcolor)
    ax_P     = fig5.add_axes([0.10,0.72,0.35,0.06], facecolor=axcolor)
    ax_xbar  = fig5.add_axes([0.10,0.64,0.35,0.06], facecolor=axcolor)
    ax_delx  = fig5.add_axes([0.10,0.56,0.35,0.06], facecolor=axcolor)
    ax_V0    = fig5.add_axes([0.55,0.80,0.35,0.06], facecolor=axcolor)
    ax_a     = fig5.add_axes([0.55,0.72,0.35,0.06], facecolor=axcolor)
    ax_alpha = fig5.add_axes([0.55,0.64,0.35,0.06], facecolor=axcolor)
    ax_m     = fig5.add_axes([0.55,0.56,0.35,0.06], facecolor=axcolor)
    ax_Lag   = fig5.add_axes([0.10,0.44,0.80,0.06], facecolor=axcolor)
    ax_Xwin  = fig5.add_axes([0.10,0.36,0.80,0.06], facecolor=axcolor)

    s_T     = Slider(ax_T,    r"$T\ /\ \mathrm{K}$",    10.0, 5000.0,  valinit=s.T,      valstep=1.0)
    s_P     = Slider(ax_P,    r"$P$",        2,    256,     valinit=s.P,      valstep=1.0)
    s_xbar  = Slider(ax_xbar, r"$\bar{x}$",       -2.0, 2.0,     valinit=s.x_bar,  valstep=0.01)
    s_delx  = Slider(ax_delx, r"$\delta x$",       0.0,  2.0,     valinit=s.delta_x,valstep=0.01)
    s_V0    = Slider(ax_V0,   r"$V_0\ /\ \mathrm{eV}$",  0.01, 2.0,     valinit=s.V0_eV,  valstep=0.001)
    s_a     = Slider(ax_a,    r"$a$",        0.05, 5.0,     valinit=s.a,      valstep=0.01)
    s_alpha = Slider(ax_alpha,r"$\alpha$",    0.05, 5.0,     valinit=s.alpha,  valstep=0.01)
    s_m     = Slider(ax_m,    r"$m$",        1.0,  5000.0,  valinit=s.m,      valstep=1.0)
    s_Lag   = Slider(ax_Lag,  "ACF max lag", 50, 10000, valinit=s.acf_max_lag, valstep=10)
    s_Xwin  = Slider(ax_Xwin, "X window (samples)", 100, 200000, valinit=s.display_window, valstep=50)

    # Buttons
    ax_apply = fig5.add_axes([0.10, 0.18, 0.18, 0.08])
    ax_pause = fig5.add_axes([0.32, 0.18, 0.18, 0.08])
    ax_reset = fig5.add_axes([0.54, 0.18, 0.18, 0.08])
    b_apply = Button(ax_apply, "Apply Params")
    b_pause = Button(ax_pause, "Pause/Resume")
    b_reset = Button(ax_reset, "Reset")

    def on_apply(evt):
        s.apply_params(s_T.val, int(s_P.val), s_xbar.val, s_delx.val, s_m.val, s_V0.val, s_a.val, s_alpha.val)
    def on_pause(evt):
        running = s.toggle_pause(); b_pause.label.set_text("Pause" if running else "Resume")
    def on_reset(evt):
        s.request_reset()
    def on_lag(val):
        s.set_acf_maxlag(val)
    def on_xwin(val):
        s.set_display_window(val)

    b_apply.on_clicked(on_apply); b_pause.on_clicked(on_pause); b_reset.on_clicked(on_reset)
    s_Lag.on_changed(on_lag); s_Xwin.on_changed(on_xwin)

    worker = threading.Thread(target=s.run, daemon=True); worker.start()

    def refresh(_evt):
        # Helper: last W points, relative x axis (1..W)
        W = s.display_window
        def tail_xy(y):
            n = len(y)
            if n == 0:
                return np.array([]), np.array([])
            start = max(0, n - int(W))
            yv = np.asarray(y)[start:]
            x  = np.arange(1, yv.size+1)
            return x, yv

        # Plot 1: C_ff/C_dd running mean (last W) + corr-aware SEM band
        y_cff = np.fromiter(s.history_cff, dtype=float, count=len(s.history_cff))
        x, yv = tail_xy(y_cff)
        line_cff.set_data(x, yv)
        if x.size > 5:
            ax1.set_xlim(x.min(), x.max())
            ymin, ymax = yv.min(), yv.max()
            if ymin==ymax: ymin-=1e-6; ymax+=1e-6
            ax1.set_ylim(ymin-0.05*abs(ymin), ymax+0.05*abs(ymax))

        # Corr-aware SEM using ACF of recent raw samples buffer
        data = np.fromiter(s.samples_cff, dtype=float, count=len(s.samples_cff))
        sem_text = "n/a"
        if data.size > 10:
            rho = acf_fft(data)
            tau = tau_int_from_acf(rho, max_lag=s.acf_max_lag)
            var_window = np.var(data, ddof=1) if data.size > 1 else 0.0
            N_total = s.stats_cff.n
            sem_eff = np.sqrt(var_window * (2.0*tau) / max(N_total,1))
            sem_text = fr"${sem_eff:.2g}$ ($\tau_{{\mathrm{{int}}}}\approx {tau:.1f}, N_{{\mathrm{{eff}}}}\approx{int(N_total/max(1.0,2.0*tau))}$)"
            # Shade mean ± sem_eff
            mean = s.stats_cff.mean
            if sem_band[0] is not None:
                sem_band[0].remove(); sem_band[0] = None
            if x.size > 1:
                sem_band[0] = ax1.fill_between(x, mean-sem_eff, mean+sem_eff, alpha=0.2)

        N = s.stats_cff.n
        mean = s.stats_cff.mean
        txt1.set_text(
            rf"$N={N}$" + "\n" + 
            rf"$C_{{ff}}(0)/C_{{dd}}(0)={mean:.6g}$" + "\n" +
            rf"SEM(corr)={sem_text}"
        )

        # Plot 2: ΔH (last W)
        y_td = np.fromiter(s.history_dH_TD, dtype=float, count=len(s.history_dH_TD))
        y_vi = np.fromiter(s.history_dH_VI, dtype=float, count=len(s.history_dH_VI))
        x2, ytd = tail_xy(y_td)
        x2b, yvi = tail_xy(y_vi)
        line_dH_TD.set_data(x2, ytd); line_dH_VI.set_data(x2b, yvi)
        if x2.size > 5 or x2b.size > 5:
            xs = np.concatenate([x2, x2b]) if (x2.size and x2b.size) else (x2 if x2.size else x2b)
            ax2.set_xlim(xs.min(), xs.max())
            vals = np.concatenate([ytd, yvi]) if (ytd.size and yvi.size) else (ytd if ytd.size else yvi)
            if vals.size:
                ymin, ymax = vals.min(), vals.max()
                if ymin==ymax: ymin-=1e-6; ymax+=1e-6
                ax2.set_ylim(ymin-0.05*abs(ymin), ymax+0.05*abs(ymax))

        # ΔH text (global delta-method SEMs)
        mean_S_TD = s.stats_S_TD.mean; sem_S_TD = s.stats_S_TD.sem
        mean_S_VI = s.stats_S_VI.mean; sem_S_VI = s.stats_S_VI.sem
        dH_TD = np.sqrt(max(0.0, 0.5*mean_S_TD)); dH_VI = np.sqrt(max(0.0, 0.5*mean_S_VI))
        sem_dH_TD = (sem_S_TD/(4*dH_TD)) if dH_TD>0 else 0.0
        sem_dH_VI = (sem_S_VI/(4*dH_VI)) if dH_VI>0 else 0.0
        N2 = s.stats_S_TD.n
        txt2.set_text(
            rf"$N={N2}$" + "\n" +
            rf"$\Delta H_{{\mathrm{{TD}}}}={dH_TD:.6g}\ \pm\ {sem_dH_TD:.2g}$" + "\n" +
            rf"$\Delta H_{{\mathrm{{VI}}}}={dH_VI:.6g}\ \pm\ {sem_dH_VI:.2g}$"
        )

        # Plot 3: potential + bead "ring" drawn as (q_k, a + b*cos(2πk/n)) with lines connecting in bead order
        q = s.latest_q
        if q is not None:
            xmin = -4.0
            xmax =  4.0
            xs = np.linspace(xmin, xmax, 800)
            Vx = V_Asym_Eckart(xs, s.V0_eV*eV_to_Eh, s.a, s.alpha)
            line_V.set_data(xs, Vx)

            # Baseline just above barrier top
            y0 = float(np.max(Vx)) * 1.3
            n = q.size
            k = np.arange(n)
            # Cosine modulation amplitude based on potential scale
            amp = 0.20 * (np.max(Vx) - np.min(Vx) + 1e-9)
            yk = y0 + amp * np.sin(2.0*np.pi*k/n)
            xk = q.copy()

            # Connect points in bead order and close the loop
            ring_line.set_data(np.r_[xk, xk[0]], np.r_[yk, yk[0]])
            ring_scatter.set_data(xk, yk)
            center_marker.set_data([s.x_bar], [y0])

            # Axes limits
            ax3.set_xlim(xmin, xmax)
            y_min = min(np.min(Vx), y0 - 2.2*amp)
            y_max = max(np.max(Vx), y0 + 2.2*amp)
            ax3.set_ylim(y_min, y_max)

        # Plot 4: histogram of C_ff/C_dd samples
        ax4.cla()
        ax4.set_title(r"Histogram of $f_v(\mathbf{q})$")
        ax4.set_xlabel(r"$C_{ff}(0)/C_{dd}(0)$"); ax4.set_ylabel("Count")
        if data.size > 0:
            bins = int(np.clip(np.sqrt(data.size), 20, 200))
            ax4.hist(data, bins=bins)
            ax4.axvline(s.stats_cff.mean, linestyle='--')
        fig4.canvas.draw_idle()

        fig1.canvas.draw_idle(); fig2.canvas.draw_idle(); fig3.canvas.draw_idle()

    # timers
    for fig in (fig1, fig2, fig3, fig4, fig5):
        fig.canvas.new_timer(interval=250).add_callback(refresh, None)
    # start separate timers to ensure refresh actually runs
    t1 = fig1.canvas.new_timer(interval=250); t1.add_callback(refresh, None); t1.start()
    t2 = fig2.canvas.new_timer(interval=250); t2.add_callback(refresh, None); t2.start()
    t3 = fig3.canvas.new_timer(interval=250); t3.add_callback(refresh, None); t3.start()
    t4 = fig4.canvas.new_timer(interval=400); t4.add_callback(refresh, None); t4.start()
    t5 = fig5.canvas.new_timer(interval=400); t5.add_callback(refresh, None); t5.start()

    def on_close(_e): s.stop()
    for fig in (fig1, fig2, fig3, fig4, fig5):
        fig.canvas.mpl_connect('close_event', on_close)
    plt.show()

if __name__ == "__main__":
    main()
