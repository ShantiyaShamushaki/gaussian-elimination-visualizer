from src.gui import LDUVisualizer
import tkinter as tk


def main():
    root = tk.Tk()
    app = LDUVisualizer(root)
    root.mainloop()
    
    
if __name__ == "__main__":
    main()
