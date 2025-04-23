import tkinter as tk
from tkinter import ttk

class FunctionalCore:
    def __init__(self, root, units):
        self.units = units
        self.root = root
        self.root.title("Functional Core")
        self.root.geometry("600x400")

        self.tab_control = ttk.Notebook(root)
        self.tab_control.pack(expand=1, fill="both")

        for unit in units:
            tab = tk.Frame(self.tab_control)
            unit.setup(tab)
            self.tab_control.add(tab, text=unit.title)
