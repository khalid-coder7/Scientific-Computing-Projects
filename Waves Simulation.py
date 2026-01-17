import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

# --- 1. PRO STYLE CONFIGURATION ---
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",     # 'Computer Modern' (LaTeX style math)
    "axes.labelsize": 10,
    "axes.titlesize": 14,
    "figure.facecolor": "#f0f2f5",
    "axes.facecolor": "#f0f2f5",
})

# Performance Settings
RES_2D = 35          # Surface Grid (35x35 is optimal for smooth CPU rendering)
RES_1D = 150         # String Resolution
X_SPAN = 3.0
FLOOR_Z = -3.0

# Pre-compute Coordinate Grids
x_vec = np.linspace(-X_SPAN, X_SPAN, RES_2D)
y_vec = np.linspace(-X_SPAN, X_SPAN, RES_2D)
X, Y = np.meshgrid(x_vec, y_vec)

x_1d = np.linspace(-X_SPAN, X_SPAN, RES_1D)
zeros_1d = np.zeros_like(x_1d)

# Particle Tracer Points (for 1D mode)
x_beads = np.linspace(-X_SPAN*0.9, X_SPAN*0.9, 15)
zeros_beads = np.zeros_like(x_beads)

# --- 2. STABILITY HELPERS ---
class Arrow3D(FancyArrowPatch):
    """Matplotlib 3.8+ Safe 3D Arrow"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def set_data(self, xs, ys, zs):
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def safe_remove(artist):
    """Prevents crashes when removing artists that don't exist"""
    if artist:
        try:
            artist.remove()
        except (ValueError, AttributeError):
            if hasattr(artist, 'collections'):
                for c in artist.collections:
                    try: c.remove()
                    except: pass

# --- 3. THE VISUALIZER CLASS ---
class WaveVisualizer:
    def __init__(self):
        # Physics State
        self.t = 0.0
        self.mode = '2D'
        self.k = 3.0
        self.direction = 45.0
        self.vp = 1.0
        self.amp = 1.0
        self.show_vectors = True

        # Setup Figure
        self.fig = plt.figure(figsize=(12, 8))
        plt.subplots_adjust(left=0.05, bottom=0.25, right=0.75, top=0.92)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Style the 3D Box
        self.style_axes()
        
        # Initialize Artists (Empty placeholders)
        self.surf = None
        self.contours = None
        
        # 1D Artists
        self.line1d, = self.ax.plot([], [], [], color='#d62728', lw=3, zorder=10)
        self.shadow1d, = self.ax.plot([], [], [], color='gray', lw=1, alpha=0.3)
        self.beads, = self.ax.plot([], [], [], 'o', color='#1f77b4', markersize=5, alpha=0.8, zorder=11)
        
        # Vector Arrow (k)
        self.k_arrow = Arrow3D([0,0],[0,0],[0,0], mutation_scale=20, lw=2, arrowstyle="-|>", color="#1f77b4")
        self.ax.add_artist(self.k_arrow)
        
        # Info Panel
        self.info_ax = plt.axes([0.78, 0.45, 0.20, 0.25], facecolor='white')
        self.info_ax.set_axis_off()
        self.info_text = self.info_ax.text(0.05, 0.95, "", va='top', family='monospace', fontsize=9)
        
        # Controls
        self.setup_gui()

    def style_axes(self):
        self.ax.set_facecolor('white')
        # Make panes transparent
        self.ax.xaxis.set_pane_color((1,1,1,0))
        self.ax.yaxis.set_pane_color((1,1,1,0))
        self.ax.zaxis.set_pane_color((1,1,1,0))
        # Remove ticks for clean look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zlim(FLOOR_Z, 4)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel(r'$\psi$')

    def calculate_z(self, x, y, t):
        # If in 1D mode, override theta to 0 so the wave always moves along X
        current_theta = 0.0 if self.mode == '1D' else self.direction
        
        theta_rad = np.radians(current_theta)
        kx, ky = self.k * np.cos(theta_rad), self.k * np.sin(theta_rad)
        omega = self.vp * self.k
        
        # The Wave Equation: psi = A * cos(k·r - wt)
        return self.amp * np.cos(kx*x + ky*y - omega*t), kx, ky, omega

    def update(self, frame):
        self.t += 0.05
        
        if self.mode == '2D':
            # 1. Calculate Surface
            Z, kx, ky, w = self.calculate_z(X, Y, self.t)
            
            # 2. Render Surface (Remove old -> Plot new)
            safe_remove(self.surf)
            self.surf = self.ax.plot_surface(X, Y, Z, cmap='viridis', 
                                             vmin=-2, vmax=2, rstride=1, cstride=1,
                                             alpha=0.9, shade=True, antialiased=False)
            
            # 3. Render Floor Contours
            safe_remove(self.contours)
            if self.show_vectors:
                self.contours = self.ax.contour(X, Y, Z, zdir='z', offset=FLOOR_Z, 
                                                levels=[self.amp*0.9], colors=['#333333'], linewidths=1, alpha=0.4)

            # 4. Render Vector Arrow
            norm = np.sqrt(kx**2 + ky**2)
            if self.show_vectors:
                scale = 1.5
                self.k_arrow.set_data([0, (kx/norm)*scale], [0, (ky/norm)*scale], [FLOOR_Z, FLOOR_Z])
                self.k_arrow.set_visible(True)
            
            # Hide 1D Stuff
            self.line1d.set_visible(False)
            self.shadow1d.set_visible(False)
            self.beads.set_visible(False)
            
            # Title
            sign_x = "" if kx >= 0 else "-"
            sign_y = "+" if ky >= 0 else "-"
            title = rf"$\psi(x,y,t) = {self.amp:.1f} \cos({sign_x}{abs(kx):.1f}x {sign_y} {abs(ky):.1f}y - {w:.1f}t)$"

        else: # 1D Mode
            # 1. Calculate Line
            Z, k, _, w = self.calculate_z(x_1d, 0, self.t)
            Z_beads, _, _, _ = self.calculate_z(x_beads, 0, self.t)
            
            # 2. Render Line
            self.line1d.set_visible(True)
            self.line1d.set_data_3d(x_1d, zeros_1d, Z)
            
            # 3. Render Shadow
            self.shadow1d.set_visible(True)
            self.shadow1d.set_data_3d(x_1d, zeros_1d, np.full_like(x_1d, FLOOR_Z))
            
            # 4. Render Particles (Beads)
            self.beads.set_visible(True)
            self.beads.set_data_3d(x_beads, zeros_beads, Z_beads)
            
            # Cleanup 2D Stuff
            safe_remove(self.surf)
            safe_remove(self.contours)
            self.surf = None
            self.contours = None
            self.k_arrow.set_visible(False)
            
            # Title
            title = rf"$\psi(x,t) = {self.amp:.1f} \cos({self.k:.1f}x - {w:.1f}t)$"

        # Update Text & Info
        self.ax.set_title(title, fontsize=15, pad=10, color="#333333")
        self.update_info_panel(w)

    def update_info_panel(self, w):
        # Calculate derived physics
        lam = 2 * np.pi / self.k
        period = 2 * np.pi / w
        
        info = (f"WAVE PARAMETERS\n"
                f"───────────────\n"
                f"Amplitude (A): {self.amp:.2f} m\n"
                f"Wavenumber k : {self.k:.2f} rad/m\n"
                f"Direction θ  : {self.direction:.0f}°\n"
                f"Phase Vel vp : {self.vp:.2f} m/s\n\n"
                f"DERIVED PROPERTIES\n"
                f"──────────────────\n"
                f"Ang. Freq ω  : {w:.2f} rad/s\n"
                f"Wavelength λ : {lam:.2f} m\n"
                f"Period T     : {period:.2f} s")
        
        self.info_text.set_text(info)

    def setup_gui(self):
        # Modern Slider Style
        bg_color = '#e8e8e8'
        slider_color = '#555555'
        
        # Dimensions: [left, bottom, width, height]
        ax_k   = plt.axes([0.1, 0.20, 0.4, 0.03], facecolor=bg_color)
        ax_dir = plt.axes([0.1, 0.16, 0.4, 0.03], facecolor=bg_color)
        ax_vp  = plt.axes([0.1, 0.12, 0.4, 0.03], facecolor=bg_color)
        ax_amp = plt.axes([0.1, 0.08, 0.4, 0.03], facecolor=bg_color)
        
        self.s_k   = Slider(ax_k,   r'$|\vec{k}|$', 0.5, 6.0, valinit=3.0, color=slider_color)
        self.s_dir = Slider(ax_dir, r'$\theta$',    0, 360, valinit=45.0, color=slider_color)
        self.s_vp  = Slider(ax_vp,  r'$v_p$',       0.1, 3.0, valinit=1.0, color=slider_color)
        self.s_amp = Slider(ax_amp, r'$A$',         0.1, 3.0, valinit=1.0, color=slider_color)
        
        # Update Callbacks
        def update_val(val):
            self.k = self.s_k.val
            self.direction = self.s_dir.val
            self.vp = self.s_vp.val
            self.amp = self.s_amp.val
            
        for s in [self.s_k, self.s_dir, self.s_vp, self.s_amp]:
            s.on_changed(update_val)
            s.label.set_color("#333333")
            s.valtext.set_color("#333333")

        # Mode Switch
        ax_rad = plt.axes([0.6, 0.12, 0.12, 0.11], frameon=False)
        self.rad = RadioButtons(ax_rad, ('1D String', '2D Surface'), active=1, activecolor='#d62728')
        
        def change_mode(label):
            self.mode = '1D' if '1D' in label else '2D'
            
            if self.mode == '1D':
                self.ax.view_init(elev=20, azim=-90)
                self.s_dir.active = False          # Disable theta slider
                self.s_dir.ax.set_alpha(0.3)       # Visual ghosting
            else:
                self.ax.view_init(elev=35, azim=-60)
                self.s_dir.active = True           # Enable theta slider
                self.s_dir.ax.set_alpha(1.0)       # Full opacity
            
            self.fig.canvas.draw_idle()
                
        self.rad.on_clicked(change_mode)
        
        # Vector Toggle
        ax_check = plt.axes([0.75, 0.12, 0.15, 0.05], frameon=False)
        self.check = CheckButtons(ax_check, ['Show Vectors'], [True])
        def toggle_vec(label): self.show_vectors = not self.show_vectors
        self.check.on_clicked(toggle_vec)

# --- 4. RUN ---
viz = WaveVisualizer()
# Default view for 2D mode start
viz.ax.view_init(elev=35, azim=-60) 

ani = animation.FuncAnimation(viz.fig, viz.update, interval=40, blit=False, cache_frame_data=False)
plt.show()