import tkinter as tk
from core import FunctionalCore
from units.home_unit import HomeUnit


if __name__ == "__main__":
    root = tk.Tk()
    units = [
        HomeUnit()
    ]
    app = FunctionalCore(root, units)
    root.mainloop()