import tkinter as tk
from tkinter import filedialog, messagebox
from unit_base import FunctionalUnit
import os
import json
import csv

class DatumaroLabelExportUnit(FunctionalUnit):
    title = "Datumaro Label Export"
    description = "Convert Datumaro's JSON annotation to frame-state CSV."

    def setup(self, parent):
        self.json_path = ""
        self.export_path = os.getcwd()

        tk.Label(parent, text="Datumaro JSON File:").pack(anchor="w")
        self.json_entry = tk.Entry(parent, width=60)
        self.json_entry.pack()
        tk.Button(parent, text="Browse", command=self.select_json_file).pack()

        tk.Label(parent, text="Export Folder:").pack(anchor="w")
        self.export_entry = tk.Entry(parent, width=60)
        self.export_entry.insert(0, self.export_path)
        self.export_entry.pack()
        tk.Button(parent, text="Browse", command=self.select_export_folder).pack()

        self.output_name_entry = tk.Entry(parent, width=60)
        self.output_name_entry.insert(0, "frame_states.csv")
        self.output_name_entry.pack()
        tk.Label(parent, text="Output File Name (e.g., frame_states.csv)").pack(anchor="w")

        self.run_button = tk.Button(parent, text="▶ Convert", command=self.run)
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

    def select_json_file(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if path:
            self.json_path = path
            self.json_entry.delete(0, "end")
            self.json_entry.insert(0, path)

    def select_export_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.export_path = path
            self.export_entry.delete(0, "end")
            self.export_entry.insert(0, path)

    def run(self):
        annotation_file = self.json_entry.get().strip()
        output_name = self.output_name_entry.get().strip()
        export_folder = self.export_entry.get().strip()

        if not os.path.isfile(annotation_file):
            messagebox.showerror("Error", "Please select a valid JSON file.")
            return

        if not output_name.endswith(".csv"):
            messagebox.showerror("Error", "Output file name must end with .csv")
            return

        output_csv = os.path.join(export_folder, output_name)
        default_label = "0"

        try:
            with open(annotation_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            id_to_label = {i: label["name"] for i, label in enumerate(data["categories"]["label"]["labels"])}

            frame_labels = {}
            for item in data["items"]:
                frame_num = item["attr"]["frame"]
                annotations = item.get("annotations", [])
                if annotations:
                    label_id = annotations[0]["label_id"]
                    label = id_to_label.get(label_id, default_label)
                    frame_labels[frame_num] = label

            total_frames = max(frame_labels.keys()) + 1

            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "state"])
                for i in range(total_frames):
                    label = frame_labels.get(i, default_label)
                    writer.writerow([i, label])

            self.status_label.config(text="✅ Conversion complete!", fg="green")
            self.log(f"✅ CSV saved to:\n{output_csv}")
        except Exception as e:
            self.status_label.config(text="❌ Error during conversion", fg="red")
            self.log(f"❌ Error: {str(e)}")