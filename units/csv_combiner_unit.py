import tkinter as tk
from tkinter import filedialog, messagebox
from unit_base import FunctionalUnit
import csv
import os

class CSVCombinerUnit(FunctionalUnit):
    title = "CSV Combiner"
    description = "Combine two CSV files row-wise with a frame gap, and label the gap states as 0."

    def setup(self, parent):
        self.file1_path = ""
        self.file2_path = ""
        self.export_path = os.getcwd()

        tk.Label(parent, text="CSV File 1:").pack(anchor="w")
        self.file1_entry = tk.Entry(parent, width=60)
        self.file1_entry.pack()
        tk.Button(parent, text="Browse", command=self.select_file1).pack()

        tk.Label(parent, text="CSV File 2:").pack(anchor="w")
        self.file2_entry = tk.Entry(parent, width=60)
        self.file2_entry.pack()
        tk.Button(parent, text="Browse", command=self.select_file2).pack()

        tk.Label(parent, text="Export Folder:").pack(anchor="w")
        self.export_entry = tk.Entry(parent, width=60)
        self.export_entry.insert(0, self.export_path)
        self.export_entry.pack()
        tk.Button(parent, text="Browse", command=self.select_export_folder).pack()

        tk.Label(parent, text="Output File Name:").pack(anchor="w")
        self.output_name_entry = tk.Entry(parent, width=60)
        self.output_name_entry.insert(0, "combined_output.csv")
        self.output_name_entry.pack()

        tk.Label(parent, text="Gap Size (frames with state '0'):").pack(anchor="w")
        self.gap_size_entry = tk.Entry(parent, width=10)
        self.gap_size_entry.insert(0, "5")
        self.gap_size_entry.pack()

        self.run_button = tk.Button(parent, text="▶ Combine", command=self.run)
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

    def select_file1(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.file1_path = path
            self.file1_entry.delete(0, "end")
            self.file1_entry.insert(0, path)

    def select_file2(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.file2_path = path
            self.file2_entry.delete(0, "end")
            self.file2_entry.insert(0, path)

    def select_export_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.export_path = path
            self.export_entry.delete(0, "end")
            self.export_entry.insert(0, path)

    def run(self):
        file1 = self.file1_entry.get().strip()
        file2 = self.file2_entry.get().strip()
        output_name = self.output_name_entry.get().strip()
        export_path = self.export_entry.get().strip()

        try:
            gap_size = int(self.gap_size_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Gap size must be a valid integer.")
            return

        if not os.path.isfile(file1) or not os.path.isfile(file2):
            messagebox.showerror("Error", "Please select two valid CSV files.")
            return

        if not output_name.endswith(".csv"):
            messagebox.showerror("Error", "Output file name must end with .csv")
            return

        output_file = os.path.join(export_path, output_name)

        try:
            with open(file1, newline='', encoding='utf-8') as f1, \
                 open(file2, newline='', encoding='utf-8') as f2, \
                 open(output_file, 'w', newline='', encoding='utf-8') as out:

                reader1 = list(csv.reader(f1))
                reader2 = list(csv.reader(f2))
                writer = csv.writer(out)

                header = reader1[0]
                writer.writerow(header)

                data1 = reader1[1:]
                data2 = reader2[1:]

                frame_col_index = header.index("frame") if "frame" in header else 0
                state_col_index = header.index("state") if "state" in header else None

                # write all from file 1
                writer.writerows(data1)
                last_frame_index = int(data1[-1][frame_col_index]) if data1 else 0

                # create gap frames with state = "0"
                for i in range(1, gap_size + 1):
                    row = data1[-1].copy()
                    row[frame_col_index] = str(last_frame_index + i)
                    if state_col_index is not None:
                        row[state_col_index] = "0"
                    writer.writerow(row)

                # write adjusted file 2 data
                for i, row in enumerate(data2):
                    row = row.copy()
                    row[frame_col_index] = str(last_frame_index + gap_size + i + 1)
                    if state_col_index is not None and i < gap_size:
                        row[state_col_index] = "0"
                    writer.writerow(row)

            self.status_label.config(text="✅ Combine complete!", fg="green")
            self.log(f"✅ Combined CSV saved to:\n{output_file}")
        except Exception as e:
            self.status_label.config(text="❌ Error during combine", fg="red")
            self.log(f"❌ Error: {str(e)}")