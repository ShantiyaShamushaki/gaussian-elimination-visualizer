# gui_ldu_stepper.py
import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .solver import LDU_factorize

class LDUVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Gaussian LDU Step Visualizer")
        self.root.geometry("1200x650")
        self.root.configure(bg="#f4f4f4")

        # --- Input area ---
        tk.Label(root, text="Matrix (comma-separated rows):",
                 font=("Consolas", 11, "bold"), bg="#f4f4f4").pack(pady=8)
        self.text_area = tk.Text(root, width=50, height=6, font=("Consolas", 11))
        self.text_area.insert(tk.END, "4,3,2\n3,2,1\n2,1,3")
        self.text_area.pack(pady=3)

        tk.Button(root, text="Run LDU Factorization", bg="#4CAF50", fg="white",
                  font=("Consolas", 11, "bold"), command=self.run_ldu).pack(pady=10)

        # --- Canvas area for plots ---
        self.fig, self.axes = plt.subplots(1, 3, figsize=(10, 3))
        plt.subplots_adjust(wspace=0.4)
        for ax, title in zip(self.axes, ['L', 'D', 'U']):
            ax.set_title(title)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=10)

        # --- Step control buttons ---
        control_frame = tk.Frame(root, bg="#f4f4f4")
        control_frame.pack()

        self.prev_btn = tk.Button(control_frame, text="⬅ Prev", width=10,
                                  command=self.prev_step, state="disabled")
        self.prev_btn.grid(row=0, column=0, padx=10)

        self.next_btn = tk.Button(control_frame, text="Next ➡", width=10,
                                  command=self.next_step, state="disabled")
        self.next_btn.grid(row=0, column=1, padx=10)

        # --- Status ---
        self.status = tk.Label(root, text="Load Matrix & Run", bg="#f4f4f4",
                               font=("Consolas", 10))
        self.status.pack(pady=8)
        self.operation_label = tk.Label(root, text="Operation: ---",
                                bg="#f4f4f4", font=("Consolas", 11))
        self.operation_label.pack(pady=4)

        # --- state ---
        self.L, self.D, self.U, self.log = None, None, None, None
        self.cur_idx = 0
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        plt.close('all')      # Close all matplotlib windows
        self.fig.clear()      # Release figure memory
        self.root.destroy()   

    # =======================================================
    def run_ldu(self):
        try:
            mat_text = self.text_area.get("1.0", tk.END).strip()
            rows = [list(map(float, r.split(','))) for r in mat_text.split('\n') if r.strip()]
            A = np.array(rows)

            self.L, self.D, self.U, self.log = LDU_factorize(A)
            self.cur_idx = 0
            self.next_btn.config(state="normal")
            self.prev_btn.config(state="disabled")
            self.update_display()
            self.status.config(text=f"Matrix size: {A.shape[0]}×{A.shape[1]} | Steps: {len(self.log)}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def render_matrix(self, ax, M, name):
        ax.clear()
        im = ax.imshow(M, cmap='coolwarm', interpolation='none', origin='upper')
        ax.set_title(name)
        for (i, j), val in np.ndenumerate(M):
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontsize=8)
        return im

    def update_display(self):
        tag, _, explanation = self.log[self.cur_idx]

        # Construct matrices up to this step
        L_cur = np.identity(self.L.shape[0])
        D_cur = np.identity(self.D.shape[0])
        U_cur = np.identity(self.U.shape[0])

        for t, mat, _ in self.log[:self.cur_idx+1]:
            if t == "L":
                L_cur = mat.copy()
            elif t == "D":
                D_cur = mat.copy()
            elif t == "U" or t == "INIT":
                U_cur = mat.copy()

        # Draw matrices
        self.render_matrix(self.axes[0], L_cur, "L")
        self.render_matrix(self.axes[1], D_cur, "D")
        self.render_matrix(self.axes[2], U_cur, "U")
        self.canvas.draw()

        # ✅ NEW: show explanation
        self.operation_label.config(text=f"Operation: {explanation}")

        self.status.config(text=f"Step {self.cur_idx+1}/{len(self.log)}")
    
    # =======================================================
    def next_step(self):
        if self.cur_idx < len(self.log) - 1:
            self.cur_idx += 1
            self.update_display()
            self.prev_btn.config(state="normal")
        if self.cur_idx == len(self.log) - 1:
            self.next_btn.config(state="disabled")

    def prev_step(self):
        if self.cur_idx > 0:
            self.cur_idx -= 1
            self.update_display()
            self.next_btn.config(state="normal")
        if self.cur_idx == 0:
            self.prev_btn.config(state="disabled")

