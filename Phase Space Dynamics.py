import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox, Slider, CheckButtons
from matplotlib.lines import Line2D
import sympy as sp
from scipy.optimize import fsolve

# --- 1. SETTINGS & STYLING ---
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'

def safe_parse(latex_str):
    try:
        x, y, t = sp.symbols('x y t', real=True)
        locals_dict = {'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 
                       'tan': sp.tan, 'log': sp.log, 'sqrt': sp.sqrt, 'pi': sp.pi}
        expr = sp.sympify(latex_str, locals=locals_dict)
        func = sp.lambdify((x, y, t), expr, modules=['numpy'])
        return func, expr, sp.latex(expr)
    except:
        return None, None, r"\text{Error}"

class PhaseSpaceExplorer:
    def __init__(self):
        self.dt = 0.04
        self.t_arr = np.arange(0, 40.0, self.dt)
        self.is_running = False
        self.sim_idx = 0
        
        self.params = {'x0': 1.0, 'y0': 1.0, 'dx': "y", 'dy': "x - x**3 - 0.2*y", 'speed': 1.0}
        self.show_manifolds = True
        self.show_nullclines = True
        
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = self.fig.add_axes([0.1, 0.25, 0.60, 0.65])
        self.eq_text = self.fig.text(0.1, 0.93, "", fontsize=15, color='#2c3e50')
        
        self.line_faint, = self.ax.plot([], [], '-', lw=1.2, color='gray', alpha=0.2, zorder=1)
        self.line_traj, = self.ax.plot([], [], '-', lw=2.2, color='#3498db', zorder=5)
        self.point, = self.ax.plot([], [], 'ro', ms=7, zorder=10, mec='white')
        
        self.hud = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                verticalalignment='top', fontsize=10, family='serif',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        self.quiver = None
        self.topo_artists = []      # Fixed points
        self.manifold_artists = []  # Ws and Wu
        self.nullcline_artists = [] # Contours
        
        self.setup_ui()
        self.create_legend()
        self.update_system_logic(self.params['dx'], self.params['dy'])
        
        self.ani = FuncAnimation(self.fig, self.update_frame, interval=25, cache_frame_data=False)
        plt.show()

    def setup_ui(self):
        # Coordinates and Dynamics
        self.tx_dx = TextBox(plt.axes([0.1, 0.12, 0.12, 0.04]), r'$\dot{x}=$ ', initial=self.params['dx'])
        self.tx_dy = TextBox(plt.axes([0.28, 0.12, 0.12, 0.04]), r'$\dot{y}=$ ', initial=self.params['dy'])
        self.tx_x0 = TextBox(plt.axes([0.1, 0.06, 0.05, 0.04]), r'$x_0=$ ', initial=str(self.params['x0']))
        self.tx_y0 = TextBox(plt.axes([0.2, 0.06, 0.05, 0.04]), r'$y_0=$ ', initial=str(self.params['y0']))

        # Toggles
        rax = plt.axes([0.45, 0.05, 0.12, 0.08], frameon=False)
        self.check = CheckButtons(rax, ['Manifolds', 'Nullclines'], [True, True])
        self.check.on_clicked(self.toggle_visibility)

        # Speed and Buttons
        self.sld_speed = Slider(plt.axes([0.1, 0.01, 0.15, 0.03]), r'$\text{Speed}$ ', 0.1, 5.0, valinit=1.0)
        self.btn_play = Button(plt.axes([0.6, 0.12, 0.06, 0.04]), r'$\text{Play}$', color='#e8f5e9')
        self.btn_stop = Button(plt.axes([0.67, 0.12, 0.06, 0.04]), r'$\text{Stop}$', color='#ffebee')
        self.btn_rst = Button(plt.axes([0.6, 0.06, 0.13, 0.04]), r'$\text{Reset}$', color='#e3f2fd')

        # Bindings
        self.tx_dx.on_submit(lambda v: self.update_system_logic(v, self.tx_dy.text))
        self.tx_dy.on_submit(lambda v: self.update_system_logic(self.tx_dx.text, v))
        self.tx_x0.on_submit(self.update_coords_logic)
        self.tx_y0.on_submit(self.update_coords_logic)
        self.sld_speed.on_changed(lambda v: self.params.update({'speed': v}))
        self.btn_play.on_clicked(lambda e: setattr(self, 'is_running', True))
        self.btn_stop.on_clicked(lambda e: setattr(self, 'is_running', False))
        self.btn_rst.on_clicked(lambda e: self.reset_trajectory())

    def toggle_visibility(self, label):
        if label == 'Manifolds': self.show_manifolds = not self.show_manifolds
        if label == 'Nullclines': self.show_nullclines = not self.show_nullclines
        self.refresh_topology()

    def update_system_logic(self, dx_str, dy_str):
        fdx, ex, lx = safe_parse(dx_str)
        fdy, ey, ly = safe_parse(dy_str)
        if fdx and fdy:
            self.f_dx, self.f_dy, self.e_x, self.e_y = fdx, fdy, ex, ey
            self.params['dx'], self.params['dy'] = dx_str, dy_str
            self.eq_text.set_text(fr"$\mathbf{{System:}} \quad \dot{{x}} = {lx}, \quad \dot{{y}} = {ly}$")
            x, y = sp.symbols('x y')
            self.jac_expr = sp.Matrix([self.e_x, self.e_y]).jacobian([x, y])
            self.reset_trajectory()
            self.refresh_topology()

    def update_coords_logic(self, _=None):
        try:
            self.params['x0'] = float(self.tx_x0.text)
            self.params['y0'] = float(self.tx_y0.text)
            self.reset_trajectory()
        except: pass

    def reset_trajectory(self):
        self.is_running = False
        self.sim_idx = 0
        try:
            cx, cy = float(self.params['x0']), float(self.params['y0'])
            x_vals, y_vals = [cx], [cy]
            dt = self.dt
            
            for i in range(len(self.t_arr)):
                t = self.t_arr[i]
                
                # k1: Start point
                k1x = self.f_dx(cx, cy, t)
                k1y = self.f_dy(cx, cy, t)
                
                # k2: Midpoint using k1
                k2x = self.f_dx(cx + 0.5*dt*k1x, cy + 0.5*dt*k1y, t + 0.5*dt)
                k2y = self.f_dy(cx + 0.5*dt*k1x, cy + 0.5*dt*k1y, t + 0.5*dt)
                
                # k3: Midpoint using k2
                k3x = self.f_dx(cx + 0.5*dt*k2x, cy + 0.5*dt*k2y, t + 0.5*dt)
                k3y = self.f_dy(cx + 0.5*dt*k2x, cy + 0.5*dt*k2y, t + 0.5*dt)
                
                # k4: End point using k3
                k4x = self.f_dx(cx + dt*k3x, cy + dt*k3y, t + dt)
                k4y = self.f_dy(cx + dt*k3x, cy + dt*k3y, t + dt)
                
                # Weighted average update
                cx += (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
                cy += (dt/6.0) * (k1y + 2*k2y + 2*k3y + k4y)
                
                if not (np.isfinite(cx) and np.isfinite(cy)): break
                if abs(cx) > 1e4 or abs(cy) > 1e4: break
                    
                x_vals.append(cx)
                y_vals.append(cy)
                
            self.sol_x, self.sol_y = np.array(x_vals), np.array(y_vals)
        except:
            self.sol_x = np.array([float(self.params['x0'])])
            self.sol_y = np.array([float(self.params['y0'])])

        self.line_faint.set_data(self.sol_x, self.sol_y)
        self.line_traj.set_data([], [])
        self.point.set_data([self.sol_x[0]], [self.sol_y[0]])
        self.update_hud(0)
        self.fig.canvas.draw_idle()

    def refresh_topology(self):
        # 1. Clear all old topology artists
        for a in self.topo_artists + self.manifold_artists:
            try: a.remove()
            except: pass
        self.topo_artists, self.manifold_artists = [], []
        
        for nc in self.nullcline_artists:
            try:
                for coll in nc.collections: coll.remove()
            except AttributeError:
                nc.remove() # Handle newer Matplotlib versions
        self.nullcline_artists = []

        # 2. Dynamic Bounds
        max_val = max(np.max(np.abs(self.sol_x)), np.max(np.abs(self.sol_y)), 2.0)
        lim = max_val * 1.4
        self.ax.set_xlim(-lim, lim); self.ax.set_ylim(-lim, lim)

        # 3. Vector Field
        if self.quiver is not None:
            try:
                self.quiver.remove()
            except (ValueError, RuntimeError):
                pass 
        
        X, Y = np.meshgrid(np.linspace(-lim, lim, 20), np.linspace(-lim, lim, 20))
        
        # 💡 FIX: Force conversion to float to prevent SymPy/NumPy conflicts
        # Use a list comprehension or np.vectorize to handle potential scalar outputs
        u_func = np.vectorize(lambda x, y: float(self.f_dx(x, y, 0)))
        v_func = np.vectorize(lambda x, y: float(self.f_dy(x, y, 0)))
        
        U = u_func(X, Y)
        V = v_func(X, Y)
        
        M = np.hypot(U, V)
        M = np.where(M == 0, 1, M) # Prevent division by zero
        
        self.quiver = self.ax.quiver(X, Y, U/M, V/M, color='#bdc3c7', alpha=0.3, pivot='mid', scale=25)

        # 4. Nullclines
        if self.show_nullclines:
            Xn, Yn = np.meshgrid(np.linspace(-lim, lim, 100), np.linspace(-lim, lim, 100))
            self.nullcline_artists.append(self.ax.contour(Xn, Yn, self.f_dx(Xn, Yn, 0), levels=[0], colors='green', alpha=0.4, linestyles=':'))
            self.nullcline_artists.append(self.ax.contour(Xn, Yn, self.f_dy(Xn, Yn, 0), levels=[0], colors='orange', alpha=0.4, linestyles=':'))

        # 5. Fixed Points & Manifolds
        seeds = np.linspace(-lim, lim, 6)
        found_pts = []
        for sx in seeds:
            for sy in seeds:
                sol, _, ier, _ = fsolve(lambda p: [self.f_dx(p[0],p[1],0), self.f_dy(p[0],p[1],0)], [sx, sy], full_output=True)
                if ier == 1:
                    pt = np.round(sol, 4)
                    if not any(np.allclose(pt, p, atol=1e-2) for p in found_pts):
                        found_pts.append(pt)
                        self.process_equilibrium(pt, lim)
        self.fig.canvas.draw_idle()

    def process_equilibrium(self, pt, lim):
        x_s, y_s = sp.symbols('x y')
        J = np.array(self.jac_expr.subs({x_s:pt[0], y_s:pt[1], sp.Symbol('t'):0}).tolist(), dtype=float)
        evs, evecs = np.linalg.eig(J)
        re = np.real(evs)
        is_saddle = (re[0] * re[1] < -1e-5)
        color = 'black' if is_saddle else ('blue' if all(re < -1e-5) else 'red')
        
        pm, = self.ax.plot(pt[0], pt[1], 'x' if is_saddle else 'o', color=color, ms=10, mew=2, zorder=15, mfc='none' if color=='red' else color)
        self.topo_artists.append(pm)
        
        if is_saddle and self.show_manifolds:
            eps = 1e-3
            for i in range(2):
                vec = np.real(evecs[:, i])
                for sign in [1, -1]:
                    path = self.integrate_manifold(pt + sign*eps*vec, lim, forward=(re[i] > 0))
                    l, = self.ax.plot(path[:,0], path[:,1], 'r--' if re[i]>0 else 'b-', lw=1.2, alpha=0.6, zorder=2)
                    self.manifold_artists.append(l)

    def integrate_manifold(self, start, lim, forward=True):
        curr = np.array(start, dtype=float)
        path = [curr.copy()]
        dt_m = 0.05 if forward else -0.05
        for _ in range(1000):
            vx, vy = self.f_dx(curr[0], curr[1], 0), self.f_dy(curr[0], curr[1], 0)
            curr += np.array([vx, vy]) * dt_m
            path.append(curr.copy())
            if np.linalg.norm(curr) > lim * 2.0: break
        return np.array(path)

    def create_legend(self):
        ax_leg = self.fig.add_axes([0.72, 0.25, 0.25, 0.65])
        ax_leg.axis('off')
        els = [
            Line2D([0], [0], marker='o', color='w', mfc='blue', label=r'$\text{Stable (Sink)}$'),
            Line2D([0], [0], marker='o', color='w', mfc='none', mec='red', mew=1.5, label=r'$\text{Unstable (Source)}$'),
            Line2D([0], [0], marker='x', color='black', mew=2, label=r'$\text{Saddle Point}$'),
            Line2D([0], [0], color='blue', lw=1.5, label=r'$W^s \text{ (Stable)}$'),
            Line2D([0], [0], color='red', lw=1.5, ls='--', label=r'$W^u \text{ (Unstable)}$'),
            Line2D([0], [0], color='green', lw=1, ls=':', label=r'$\dot{x}=0$'),
            Line2D([0], [0], color='orange', lw=1, ls=':', label=r'$\dot{y}=0$')
        ]
        ax_leg.legend(handles=els, loc='upper left', title=r"$\mathbf{Legend}$", fontsize=9)

    def update_hud(self, idx):
        self.hud.set_text(fr"$\text{{t}}: {self.t_arr[idx]:.2f}$" + "\n" + 
                           fr"$\text{{x}}: {self.sol_x[idx]:.3f}$" + "\n" + 
                           fr"$\text{{y}}: {self.sol_y[idx]:.3f}$")

    def update_frame(self, frame):
        if not self.is_running: return self.line_traj, self.point, self.hud
        step = max(1, int(self.params['speed']))
        if self.sim_idx < len(self.sol_x) - step:
            self.sim_idx += step
            self.line_traj.set_data(self.sol_x[:self.sim_idx], self.sol_y[:self.sim_idx])
            self.point.set_data([self.sol_x[self.sim_idx]], [self.sol_y[self.sim_idx]])
            self.update_hud(self.sim_idx)
        else:
            self.is_running = False
        return self.line_traj, self.point, self.hud

if __name__ == "__main__":
    PhaseSpaceExplorer()