import tkinter as tk
from core import FunctionalCore
from units.home_unit import HomeUnit
from units.mediapipe_data_generator_unit import MediaPipeDataGeneratorUnit
from units.datumaro_label_export_unit import DatumaroLabelExportUnit
from units.csv_merger_unit import CSVMergerUnit


if __name__ == "__main__":
    root = tk.Tk()
    units = [
        HomeUnit(),
        MediaPipeDataGeneratorUnit(),
        DatumaroLabelExportUnit(),
        CSVMergerUnit()
    ]
    app = FunctionalCore(root, units)
    root.mainloop()