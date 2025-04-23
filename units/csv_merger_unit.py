import tkinter as tk
from tkinter import filedialog, messagebox
from unit_base import FunctionalUnit
import csv
import os

class CSVMergerUnit(FunctionalUnit):
    title = "CSV Merger"
    description = "Merge two CSV files while avoiding duplicate columns."

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

        self.output_name_entry = tk.Entry(parent, width=60)
        self.output_name_entry.insert(0, "merged_output.csv")
        self.output_name_entry.pack()
        tk.Label(parent, text="Output File Name (e.g., merged_output.csv)").pack(anchor="w")

        self.run_button = tk.Button(parent, text="▶ Merge", command=self.run)
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

                reader1 = csv.reader(f1)
                reader2 = csv.reader(f2)
                writer = csv.writer(out)

                header1 = next(reader1)
                header2 = next(reader2)
                unique_header2 = [col for col in header2 if col not in header1]
                writer.writerow(header1 + unique_header2)

                unique_indices2 = [i for i, col in enumerate(header2) if col in unique_header2]

                for row1, row2 in zip(reader1, reader2):
                    filtered_row2 = [row2[i] for i in unique_indices2]
                    writer.writerow(row1 + filtered_row2)

            self.status_label.config(text="✅ Merge complete!", fg="green")
            self.log(f"✅ Merged CSV saved to:\n{output_file}")
        except Exception as e:
            self.status_label.config(text="❌ Error during merge", fg="red")
            self.log(f"❌ Error: {str(e)}")