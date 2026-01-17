import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

class BilliardUniversalBankSim:
    def __init__(self, root):
        self.root = root
        self.root.title("Vector-Based Bank Solver (Mirror Physics)")
        self.root.configure(bg="white")
        
        # Table & Physics
        self.W, self.H = 1.12, 2.24
        self.g = 9.81
        self.radius = 0.028
        
        self.bounds = {
            'x_min': self.radius, 'x_max': self.W - self.radius,
            'y_min': self.radius, 'y_max': self.H - self.radius
        }

        self.start_pos = np.array([0.3, 0.5])
        self.target_pos = np.array([0.8, 1.8])
        self.walls = ["Top", "Bottom", "Left", "Right"]
        
        self.dragging_start = False
        self.dragging_target = False
        self.animating = False

        self.setup_gui()
        self.refresh()

    def setup_gui(self):
        self.main_container = tk.Frame(self.root, bg="white")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(5, 7), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_container)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # SIDEBAR
        self.sidebar = tk.Frame(self.main_container, width=280, bg="#f8f9fa", padx=15, pady=20)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        
        tk.Label(self.sidebar, text="BANK SOLVER", font=("Arial", 12, "bold"), bg="#f8f9fa").pack(pady=(0, 10))
        
        # 1. WALL SLIDER
        tk.Label(self.sidebar, text="Bank Wall Selection:", bg="#f8f9fa", font=("Arial", 10, "bold")).pack(anchor="w")
        self.wall_slider = tk.Scale(self.sidebar, from_=0, to=3, orient=tk.HORIZONTAL, 
                                    showvalue=0, command=lambda x: self.refresh(), bg="#f8f9fa", bd=0)
        self.wall_slider.set(3)
        self.wall_slider.pack(fill=tk.X)
        self.wall_indicator = tk.Label(self.sidebar, text="Wall: Right", bg="#f8f9fa", fg="#2c3e50")
        self.wall_indicator.pack(anchor="w", pady=(0, 10))

        # 2. TARGET SPEED SLIDER
        tk.Label(self.sidebar, text="Target Arrival Speed:", bg="#f8f9fa", font=("Arial", 10, "bold")).pack(anchor="w")
        max_speed = 2.0
        # Initialize the scale without the command first to avoid early refresh
        self.speed_slider = ttk.Scale(self.sidebar, from_=0.0, to=max_speed, orient=tk.HORIZONTAL)
        self.speed_slider.set(max_speed/2)
        self.speed_slider.pack(fill=tk.X)
        self.speed_label = tk.Label(self.sidebar, text="0.00 m/s", bg="#f8f9fa")
        self.speed_label.pack(anchor="w", pady=(0, 10))

        # 3. STATS DISPLAY
        self.stats_label = tk.Label(self.sidebar, text="", font=("Consolas", 9), justify=tk.LEFT, bg="#f8f9fa")
        self.stats_label.pack(fill=tk.BOTH, expand=True)

        # BOTTOM CONTROLS (Friction)
        ctrl = tk.Frame(self.root, padx=10, pady=10, bg="white")
        ctrl.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(ctrl, text="▶ Play Shot", command=self.start_animation).pack(side=tk.LEFT, padx=5)
        
        tk.Label(ctrl, text="Friction (μ):", bg="white").pack(side=tk.LEFT, padx=(10, 0))
        self.mu_scale = ttk.Scale(ctrl, from_=0.01, to=0.15, orient=tk.HORIZONTAL)
        self.mu_scale.set(0.03)
        self.mu_scale.pack(side=tk.LEFT, padx=5)
        
        self.mu_label = tk.Label(ctrl, text="0.03 (0.29 m/s²)", bg="white", width=18)
        self.mu_label.pack(side=tk.LEFT)

        # NOW attach the commands after all widgets exist
        self.speed_slider.configure(command=lambda x: self.refresh())
        self.mu_scale.configure(command=lambda x: self.refresh())

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

    def calculate_physics(self):
        wall_idx = int(self.wall_slider.get())
        wall = self.walls[wall_idx]
        self.wall_indicator.config(text=f"Wall: {wall}")
        
        mu = float(self.mu_scale.get())
        accel = mu * self.g
        self.mu_label.config(text=f"{mu:.2f} ({accel:.2f} m/s²)")

        target = self.target_pos.copy()
        
        # MIRROR METHOD
        if wall == "Top":      mirror_p = np.array([target[0], 2 * self.bounds['y_max'] - target[1]])
        elif wall == "Bottom": mirror_p = np.array([target[0], 2 * self.bounds['y_min'] - target[1]])
        elif wall == "Left":   mirror_p = np.array([2 * self.bounds['x_min'] - target[0], target[1]])
        elif wall == "Right":  mirror_p = np.array([2 * self.bounds['x_max'] - target[0], target[1]])

        d_vec = mirror_p - self.start_pos
        try:
            if wall in ["Top", "Bottom"]:
                target_coord = self.bounds['y_max'] if wall == "Top" else self.bounds['y_min']
                t = (target_coord - self.start_pos[1]) / d_vec[1]
            else:
                target_coord = self.bounds['x_max'] if wall == "Right" else self.bounds['x_min']
                t = (target_coord - self.start_pos[0]) / d_vec[0]
            bounce_p = self.start_pos + t * d_vec
        except ZeroDivisionError: return None

        d1 = np.linalg.norm(bounce_p - self.start_pos)
        d2 = np.linalg.norm(self.target_pos - bounce_p)
        total_dist = d1 + d2
        
        vf_mag = float(self.speed_slider.get())
        self.speed_label.config(text=f"{vf_mag:.2f} m/s")
        
        # Calculate required initial velocity: v0 = sqrt(vf^2 + 2ad)
        v0_mag = np.sqrt(vf_mag**2 + 2 * accel * total_dist)
        
        angle_init = np.degrees(np.arctan2(d_vec[1], d_vec[0]))
        out_vec = self.target_pos - bounce_p
        angle_target = np.degrees(np.arctan2(out_vec[1], out_vec[0]))
        
        return bounce_p, v0_mag, vf_mag, angle_init, angle_target, d1, d2, accel

    def refresh(self):
        if self.animating: return
        self.ax.clear()
        
        rail = 0.08
        self.ax.add_patch(plt.Rectangle((-rail, -rail), self.W+2*rail, self.H+2*rail, color='#5d3a1a', zorder=1))
        self.ax.add_patch(plt.Rectangle((0, 0), self.W, self.H, color='#2d7a31', zorder=2))
        
        res = self.calculate_physics()
        if res:
            bounce_p, v0_mag, vf_mag, a_init, a_target, d1, d2, accel = res
            
            self.ax.plot([self.start_pos[0], bounce_p[0]], [self.start_pos[1], bounce_p[1]], 'w--', alpha=0.5, zorder=10)
            self.ax.plot([bounce_p[0], self.target_pos[0]], [bounce_p[1], self.target_pos[1]], 'w--', alpha=0.5, zorder=10)
            self.ax.plot(bounce_p[0], bounce_p[1], 'yo', ms=8, alpha=0.75, zorder=11)
            
            stats_text = (
                f"--- START ---\n"
                f"X: {self.start_pos[0]:.2f}m, Y: {self.start_pos[1]:.2f}m\n"
                f"Launch Speed:  {v0_mag:.2f} m/s\n"
                f"Launch Angle: {a_init:.1f}°\n\n"
                f"--- TARGET ---\n"
                f"X: {self.target_pos[0]:.2f}m, Y: {self.target_pos[1]:.2f}m\n"
                f"Arrival Speed: {vf_mag:.2f} m/s\n"
                f"Arrival Angle: {a_target:.1f}°\n\n"
                f"Total Path: {d1+d2:.2f}m"
            )
            self.stats_label.config(text=stats_text)
        
        self.ax.add_patch(plt.Circle(self.start_pos, 1.5*self.radius, fc='white', ec='black', zorder=20))
        self.ax.plot(self.target_pos[0], self.target_pos[1], 'rx', ms=12, mew=3, zorder=20)
        
        self.ax.set_xlim(-0.1, self.W + 0.1)
        self.ax.set_ylim(-0.1, self.H + 0.1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.canvas.draw()

    def start_animation(self):
        if self.animating: return
            
        res = self.calculate_physics()
        if not res: return
            
        self.animating = True
        bounce_p, v0_mag, vf_mag, _, _, d1, d2, accel = res
        
        fps = 50
        # Calculate total travel time based on deceleration
        discriminant = max(0, v0_mag**2 - 2 * accel * (d1 + d2))
        duration = (v0_mag - np.sqrt(discriminant)) / (accel + 1e-6)

        # Create the moving ball artist
        self.temp_ball = plt.Circle(self.start_pos, 1.5*self.radius, fc='white', ec='black', zorder=30)
        self.ax.add_patch(self.temp_ball)
        
        v1_dir = (bounce_p - self.start_pos) / (d1 + 1e-9)
        v2_dir = (self.target_pos - bounce_p) / (d2 + 1e-9)

        def animate(i):
            if not hasattr(self, 'temp_ball'): return []
            
            t = i / fps
            # Calculate distance traveled at time t
            s = v0_mag * t - 0.5 * accel * t**2
            
            # CRITICAL FIX: Clamp 's' so it cannot exceed the total distance
            # but also cannot be less than 0
            s = max(0, min(s, d1 + d2))
            
            if s <= d1:
                pos = self.start_pos + v1_dir * s
            else:
                pos = bounce_p + v2_dir * (s - d1)
                
            self.temp_ball.set_center(pos)
            return [self.temp_ball]

        # Add 10 extra frames to 'hold' the ball at the target position
        frames = int(duration * fps) + 10 
        
        self.anim_obj = animation.FuncAnimation(
            self.fig, animate, frames=frames, interval=20, 
            blit=True, repeat=False
        )
        
        # Delay cleanup by an extra half-second so you see the impact
        self.root.after(int(duration * 1000) + 600, self.end_anim)
        self.canvas.draw_idle()

    def end_anim(self):
        # Safely stop animation
        if hasattr(self, 'anim_obj') and self.anim_obj is not None:
            try: self.anim_obj.event_source.stop()
            except: pass
            self.anim_obj = None

        # Safely remove the temporary ball
        if hasattr(self, 'temp_ball'):
            try: self.temp_ball.remove()
            except: pass
            delattr(self, 'temp_ball')
            
        self.animating = False
        self.refresh() # Resets the ball to the starting point

    def on_press(self, event):
        if not event.inaxes or self.animating: return
        if np.linalg.norm(np.array([event.xdata, event.ydata]) - self.target_pos) < 0.1:
            self.dragging_target = True
        elif np.linalg.norm(np.array([event.xdata, event.ydata]) - self.start_pos) < 0.1:
            self.dragging_start = True

    def on_motion(self, event):
        if not event.inaxes or self.animating: return
        x = max(self.bounds['x_min'], min(self.bounds['x_max'], event.xdata))
        y = max(self.bounds['y_min'], min(self.bounds['y_max'], event.ydata))
        if self.dragging_target: self.target_pos = np.array([x, y])
        elif self.dragging_start: self.start_pos = np.array([x, y])
        self.refresh()

    def on_release(self, event):
        self.dragging_target = self.dragging_start = False

if __name__ == "__main__":
    root = tk.Tk()
    app = BilliardUniversalBankSim(root)
    root.mainloop()