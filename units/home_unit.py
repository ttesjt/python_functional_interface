import tkinter as tk
from unit_base import FunctionalUnit

class HomeUnit(FunctionalUnit):
    title = "Home"
    description = "A demo unit."

    def setup(self, parent):
        label = tk.Label(parent, text="🏠 Home Page", font=("Arial", 16))
        label.pack(pady=50)