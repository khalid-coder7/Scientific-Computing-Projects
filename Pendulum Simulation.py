import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import time

class PendulumSimulation:
    def __init__(self):
        # --- 1. System State & Constants ---
        self.g = 9.81
        # Physics time step (resolution of the solver)
        self.dt_physics = 0.01  
        self.t_max = 120.0  
        self.t = np.arange(0, self.t_max, self.dt_physics)
        
        # Time Management
        self.sim_time_elapsed = 0.0  # Current time in the simulation (seconds)
        self.last_real_time = None   # For calculating delta time
        
        # Initial Parameters
        self.params = {
            'theta0': np.pi / 2,
            'omega0': 0.0,
            'm': 1.0,
            'L': 1.0,
            'b': 0.5,
            'speed': 1.0
        }

        # --- 2. Plot Setup ---
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle(r"Dynamical System: $\ddot{\theta} + \frac{b}{m}\dot{\theta} + \frac{g}{L}\sin(\theta) = 0$", 
                          fontsize=16, y=0.96)

        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.35, bottom=0.30)
        
        # A. Physical Pendulum
        self.ax_pend = self.fig.add_subplot(gs[0, 0])
        self.ax_pend.set_title("Physical Simulation")
        self.ax_pend.set_aspect('equal')
        self.ax_pend.set_xlim(-2.5, 2.5)
        self.ax_pend.set_ylim(-2.5, 2.5)
        self.ax_pend.grid(True, alpha=0.3)
        self.ax_pend.set_xticks([])
        self.ax_pend.set_yticks([])

        # Pendulum Artists
        self.line_rod, = self.ax_pend.plot([], [], '-', lw=3, color='black', zorder=5)
        self.bob_circle = Circle((0, -1), 0.1, fc='firebrick', ec='black', zorder=10)
        self.ax_pend.add_patch(self.bob_circle)
        
        # Trace
        self.trace_len_sec = 1.5 # Show trace for last 1.5 simulation seconds
        self.trace_lc = LineCollection([], cmap='plasma', linewidths=2)
        self.ax_pend.add_collection(self.trace_lc)
        
        # Info Text
        self.info_text = self.ax_pend.text(0.05, 0.95, '', transform=self.ax_pend.transAxes,
                                           verticalalignment='top', 
                                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # B. Phase Space
        self.ax_phase = self.fig.add_subplot(gs[0, 1])
        self.ax_phase.set_title("Phase Space")
        self.ax_phase.set_xlabel(r"$\theta$ (rad)")
        self.ax_phase.set_ylabel(r"$\dot{\theta}$=$\omega$ (rad/s)")
        self.ax_phase.grid(True, alpha=0.3)
        
        self.quiver = None 
        self.line_phase_bg, = self.ax_phase.plot([], [], '-', lw=1, color='lightgray', alpha=0.5)
        self.line_phase, = self.ax_phase.plot([], [], '-', lw=1.5, color='royalblue')
        self.point_phase, = self.ax_phase.plot([], [], 'o', color='red', zorder=10)

        # C. Energy Plot
        self.ax_energy = self.fig.add_subplot(gs[1, :])
        self.ax_energy.set_title(r"Energy Decay ($E_{total} = T + V$)")
        self.ax_energy.set_xlabel("Simulation Time (s)")
        self.ax_energy.set_ylabel("Energy (J)")
        self.ax_energy.set_xlim(0, 30.0) # Initial view
        self.ax_energy.grid(True, alpha=0.3)
        
        self.line_energy, = self.ax_energy.plot([], [], '-', lw=1, color='black', alpha=0.5)
        self.point_energy, = self.ax_energy.plot([], [], 'o', color='black', zorder=10)

        # Group moving artists
        self.moving_artists = [
            self.line_rod, self.bob_circle, self.trace_lc, self.info_text,
            self.line_phase, self.point_phase, self.point_energy
        ]
        
        for artist in self.moving_artists:
            artist.set_animated(False)

        # --- 3. Controls ---
        self.create_sliders()
        
        # --- 4. Initialization ---
        self.calculate_physics()
        self.update_static_visuals()
        
        # --- 5. Animation Setup ---
        # Interval is just the refresh rate (how smooth it looks).
        # The physics speed is now handled by time.time() inside animate.
        self.ani = FuncAnimation(self.fig, self.animate, frames=None,
                                 interval=16, blit=True, cache_frame_data=False)
        
        self.reset_animation()

    def create_sliders(self):
        ax_color = 'lightgoldenrodyellow'
        self.sliders = {}
        
        def add_slider(label, key, val_min, val_max, y_pos, x_pos=0.10):
            ax = plt.axes([x_pos, y_pos, 0.35, 0.03], facecolor=ax_color)
            s = Slider(ax, label, val_min, val_max, valinit=self.params[key])
            s.on_changed(lambda v: self.on_param_change(key, v))
            self.sliders[key] = s

        # Row 1
        add_slider(r'$\theta(0)=\theta_0$', 'theta0', -np.pi, np.pi, 0.18, 0.10)
        add_slider(r'$\dot{\theta}(0)=\omega_0$', 'omega0', -10.0, 10.0, 0.18, 0.55)
        # Row 2
        add_slider(r'Mass $m$', 'm', 0.1, 5.0, 0.13, 0.10)
        add_slider(r'Length $L$', 'L', 0.1, 3.0, 0.13, 0.55)
        # Row 3
        add_slider(r'Damping $b$', 'b', 0.0, 2.0, 0.08, 0.10)
        add_slider('Time Scale', 'speed', 0.1, 5.0, 0.08, 0.55)

        self.btn_play = Button(plt.axes([0.40, 0.02, 0.08, 0.04]), 'Play')
        self.btn_stop = Button(plt.axes([0.50, 0.02, 0.08, 0.04]), 'Stop')
        self.btn_reset = Button(plt.axes([0.60, 0.02, 0.08, 0.04]), 'Reset')
        
        self.btn_play.on_clicked(self.play)
        self.btn_stop.on_clicked(self.stop)
        self.btn_reset.on_clicked(self.reset)

    def rk4_integration(self):
        N = len(self.t)
        y = np.zeros((N, 2))
        y[0] = [self.params['theta0'], self.params['omega0']]
        
        curr = y[0]
        dt = self.dt_physics
        b = self.params['b']
        m = self.params['m']
        L = self.params['L']
        g = self.g
        
        def derivs(state):
            th, om = state
            return np.array([om, -(b/m)*om - (g/L)*np.sin(th)])

        for i in range(1, N):
            k1 = derivs(curr)
            k2 = derivs(curr + 0.5*dt*k1)
            k3 = derivs(curr + 0.5*dt*k2)
            k4 = derivs(curr + dt*k3)
            curr = curr + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            y[i] = curr
        return y

    def calculate_physics(self):
        self.y = self.rk4_integration()
        self.theta_arr = self.y[:, 0]
        self.omega_arr = self.y[:, 1]
        
        L = self.params['L']
        self.bob_x = L * np.sin(self.theta_arr)
        self.bob_y = -L * np.cos(self.theta_arr)
        
        m = self.params['m']
        g = self.g
        self.KE = 0.5 * m * (L * self.omega_arr)**2
        self.PE = m * g * L * (1 - np.cos(self.theta_arr))
        self.E_total = self.KE + self.PE

    def update_static_visuals(self):
        L = self.params['L']
        limit = L * 1.2 + 0.5
        self.ax_pend.set_xlim(-limit, limit)
        self.ax_pend.set_ylim(-limit, limit)
        self.bob_circle.radius = 0.05 + 0.04 * self.params['m']

        max_th = np.max(np.abs(self.theta_arr)) + 0.5
        max_om = np.max(np.abs(self.omega_arr)) + 0.5
        self.ax_phase.set_xlim(-max_th, max_th)
        self.ax_phase.set_ylim(-max_om, max_om)
        
        self.line_phase_bg.set_data(self.theta_arr, self.omega_arr)
        self.update_quiver(max_th, max_om)

        max_E = np.max(self.E_total)
        self.ax_energy.set_ylim(0, max_E * 1.1 if max_E > 1e-5 else 1.0)
        
        # Redraw static fills
        for collection in list(self.ax_energy.collections):
            collection.remove()
        
        # 1. Fill Kinetic (Gap between Total and PE)
        self.ax_energy.fill_between(self.t, self.PE, self.E_total, 
                                    color='orange', alpha=0.2, label=r'Kinetic ($T=\frac{1}{2}m{L}^2{\omega}^2$)')
        # 2. Fill Potential (Gap between 0 and PE)
        self.ax_energy.fill_between(self.t, 0, self.PE, 
                                    color='green', alpha=0.2, label=r'Potential ($V=mgL\left(1-\cos{\theta}\right))$')
        
        # Update line data
        self.line_energy.set_data(self.t, self.E_total)
        
        # Add Legend (with fixed location so it doesn't jump)
        self.ax_energy.legend(loc='upper right', fontsize='small', framealpha=0.8)

    def reset_animation(self):
        self.ani.event_source.stop()
        self.sim_time_elapsed = 0.0
        self.last_real_time = None
        
        # Find index 0
        self.update_artists_at_time(0.0)
        
        # Adjust Energy Plot Window to start
        self.ax_energy.set_xlim(0, 30.0)

        for artist in self.moving_artists:
            artist.set_animated(False)
        self.fig.canvas.draw() 

    def update_quiver(self, x_lim, y_lim):
        if self.quiver: self.quiver.remove()
        grid_res = 15
        T, O = np.meshgrid(np.linspace(-x_lim, x_lim, grid_res),
                           np.linspace(-y_lim, y_lim, grid_res))
        b = self.params['b']
        m = self.params['m']
        L = self.params['L']
        g = self.g
        dT = O
        dO = -(b/m)*O - (g/L)*np.sin(T)
        M = np.hypot(dT, dO)
        M[M==0] = 1
        self.quiver = self.ax_phase.quiver(T, O, dT/M, dO/M, M, 
                                           pivot='mid', cmap='autumn', alpha=0.2, zorder=1)

    def on_param_change(self, key, val):
        self.params[key] = val
        if key == 'speed':
            # Just reset the clock anchor, don't reset sim time to 0
            self.last_real_time = time.time()
        else:
            self.calculate_physics()
            self.update_static_visuals()
            self.reset_animation()

    def play(self, event):
        for artist in self.moving_artists:
            artist.set_animated(True)
        self.fig.canvas.draw()
        
        # Start the clock
        self.last_real_time = time.time()
        self.ani.event_source.start()

    def stop(self, event):
        self.ani.event_source.stop()
        self.last_real_time = None # Stop clock

    def reset(self, event):
        self.reset_animation()

    def update_info_text(self, idx, t_val):
        txt = (fr"$t$ = {t_val:.2f} s"
               "\n"
               fr"$\theta$ = {self.theta_arr[idx]:.2f} rad"
               "\n"
               fr"$\omega$ = {self.omega_arr[idx]:.2f} rad/s"
               "\n"
               fr"$E$ = {self.E_total[idx]:.2f} J")
        self.info_text.set_text(txt)

    def update_artists_at_time(self, t_current):
        # Convert time to index
        idx = int(t_current / self.dt_physics)
        
        # Bounds check
        if idx >= len(self.t):
            idx = len(self.t) - 1
            self.stop(None) # Stop if end reached

        # Update Pendulum
        x, y = self.bob_x[idx], self.bob_y[idx]
        self.line_rod.set_data([0, x], [0, y])
        self.bob_circle.center = (x, y)
        
        # Update Trace
        trace_idx_len = int(self.trace_len_sec / self.dt_physics)
        start = max(0, idx - trace_idx_len)
        pts = np.array([self.bob_x[start:idx+1], self.bob_y[start:idx+1]]).T.reshape(-1, 1, 2)
        if len(pts) > 1:
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            self.trace_lc.set_segments(segs)
            n = len(segs)
            colors = plt.cm.plasma(np.linspace(0, 1, n))
            colors[:, 3] = np.linspace(0, 0.8, n)
            self.trace_lc.set_color(colors)
        else:
            self.trace_lc.set_segments([])
        
        # Update Phase
        self.line_phase.set_data(self.theta_arr[:idx], self.omega_arr[:idx])
        self.point_phase.set_data([self.theta_arr[idx]], [self.omega_arr[idx]])
        
        # Update Energy Point
        self.point_energy.set_data([self.t[idx]], [self.E_total[idx]])
        
        # --- FIXED: Scrolling Logic for Blit ---
        current_xlim = self.ax_energy.get_xlim()
        # If the point goes past the right edge
        if self.t[idx] >= current_xlim[1]:
            # Shift the window by 30 seconds
            new_min = current_xlim[1]
            new_max = current_xlim[1] + 30.0
            self.ax_energy.set_xlim(new_min, new_max)
            
            # CRITICAL FOR BLIT: 
            # Because the axis limits changed, the "background" (ticks, grid) 
            # is now wrong. We MUST force a full redraw so the blitter captures 
            # the new background.
            self.fig.canvas.draw()
            
        self.update_info_text(idx, self.t[idx])

    def animate(self, frame):
        # 1. Get Real Time Delta
        now = time.time()
        if self.last_real_time is None:
            self.last_real_time = now
        
        dt_real = now - self.last_real_time
        self.last_real_time = now
        
        # 2. Add to Sim Time (Sim Time = Real Time * Scale)
        self.sim_time_elapsed += dt_real * self.params['speed']
        
        # 3. Update Visuals based on new Sim Time
        self.update_artists_at_time(self.sim_time_elapsed)

        return self.moving_artists

if __name__ == "__main__":
    # 1. Force Matplotlib to use the Computer Modern font set
    plt.rcParams['mathtext.fontset'] = 'cm' 
    plt.rcParams['font.family'] = 'STIXGeneral'
    sim = PendulumSimulation()
    plt.show()