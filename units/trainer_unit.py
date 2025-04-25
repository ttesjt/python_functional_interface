import tkinter as tk
from tkinter import filedialog, messagebox
from unit_base import FunctionalUnit
import os
import subprocess
import threading

class TrainerUnit(FunctionalUnit):
    title = "Model Trainer"
    description = "Train a model using external script with dataset paths and optional parameters."

    def setup(self, parent):
        self.train_path = ""
        self.test_path = ""
        self.script_path = "extensions/train_model.py"  # assumed relative to root

        tk.Label(parent, text="Training Dataset (CSV):").pack(anchor="w")
        self.train_entry = tk.Entry(parent, width=60)
        self.train_entry.pack()
        tk.Button(parent, text="Browse", command=self.select_train_file).pack()

        tk.Label(parent, text="Testing Dataset (CSV):").pack(anchor="w")
        self.test_entry = tk.Entry(parent, width=60)
        self.test_entry.pack()
        tk.Button(parent, text="Browse", command=self.select_test_file).pack()

        tk.Label(parent, text="Additional Parameters:").pack(anchor="w")
        self.params_entry = tk.Entry(parent, width=60)
        self.params_entry.insert(0, "")
        # self.params_entry.insert(0, "--epochs 20 --batch_size 32")
        self.params_entry.pack()

        self.run_button = tk.Button(parent, text="▶ Start Training", command=self.run)
        self.run_button.pack(pady=10)

        self.status_label = tk.Label(parent, text="Status: Waiting for input...")
        self.status_label.pack()

        self.log_text = tk.Text(parent, height=10, state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        print(message)

    def select_train_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.train_path = path
            self.train_entry.delete(0, "end")
            self.train_entry.insert(0, path)

    def select_test_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.test_path = path
            self.test_entry.delete(0, "end")
            self.test_entry.insert(0, path)

    def run(self):
        train_file = self.train_entry.get().strip()
        test_file = self.test_entry.get().strip()
        extra_args = self.params_entry.get().strip()

        if not os.path.isfile(train_file) or not os.path.isfile(test_file):
            messagebox.showerror("Error", "Please select valid training and testing CSV files.")
            return

        script_path = self.script_path
        if not os.path.isfile(script_path):
            messagebox.showerror("Error", f"Training script not found:\n{script_path}")
            return

        self.status_label.config(text="🚀 Training started...", fg="blue")
        self.run_button.config(state="disabled")
        self.log(f"Running script: {script_path}")
        self.log(f"Train file: {train_file}")
        self.log(f"Test file: {test_file}")
        self.log(f"Params: {extra_args or '[none]'}")

        def run_script():
            cmd = [
                "python",
                script_path,
                "--train", train_file,
                "--test", test_file,
            ]
            if extra_args:
                cmd += extra_args.split()

            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    self.log(line.strip())
                process.wait()
                self.status_label.config(text="✅ Training complete.", fg="green")
            except Exception as e:
                self.status_label.config(text="❌ Error during training", fg="red")
                self.log(f"❌ Error: {str(e)}")
            finally:
                self.run_button.config(state="normal")

        threading.Thread(target=run_script).start()