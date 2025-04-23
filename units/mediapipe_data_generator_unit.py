import tkinter as tk
from tkinter import filedialog, messagebox
from unit_base import FunctionalUnit
import cv2
import csv
import os
import numpy as np
import mediapipe as mp

class MediaPipeDataGeneratorUnit(FunctionalUnit):
    title = "MediaPipe Data Generator"
    description = "Select a video file to process with MediaPipe and extract landmarks."

    def setup(self, parent):
        self.video_path = ""
        self.export_path = os.getcwd()

        # UI Elements
        tk.Label(parent, text="Video File:").pack(anchor="w")
        self.video_entry = tk.Entry(parent, width=60)
        self.video_entry.pack()
        tk.Button(parent, text="Browse", command=self.select_video).pack()

        self.file_status = tk.Label(parent, text="❌ No file selected", fg="red")
        self.file_status.pack()

        tk.Label(parent, text="Export Folder:").pack(anchor="w")
        self.export_entry = tk.Entry(parent, width=60)
        self.export_entry.insert(0, self.export_path)
        self.export_entry.pack()
        tk.Button(parent, text="Browse", command=self.select_export_folder).pack()

        self.run_button = tk.Button(parent, text="▶ Start", state="disabled", command=self.run)
        self.run_button.pack(pady=10)

        self.progress_label = tk.Label(parent, text="Status: Waiting for input...")
        self.progress_label.pack()

        self.log_text = tk.Text(parent, height=12, state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        print(message)

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            self.video_entry.delete(0, "end")
            self.video_entry.insert(0, path)
            self.file_status.config(text=f"✅ File selected: {os.path.basename(path)}", fg="green")
            self.run_button.config(state="normal")

    def select_export_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.export_path = path
            self.export_entry.delete(0, "end")
            self.export_entry.insert(0, path)

    def run(self):
        input_video_path = self.video_entry.get().strip()
        export_path = self.export_entry.get().strip()

        if not os.path.isfile(input_video_path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        self.progress_label.config(text="Status: Running...", fg="blue")
        self.run_button.config(state="disabled")

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands

        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)

        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_video_path = os.path.join(export_path, f"{base_name}_annotated.avi")
        output_csv_path = os.path.join(export_path, f"{base_name}_landmarks.csv")

        self.log(f"🔍 Processing: {input_video_path}")
        self.log(f"📤 Exporting to: {export_path}")

        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

        selected_pose_indices = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16, 24, 23]
        csv_headers = ["frame"]
        csv_headers += [f"pose_{i}_x" for i in selected_pose_indices]
        csv_headers += [f"pose_{i}_y" for i in selected_pose_indices]
        csv_headers += ["left_hand_area", "right_hand_area"]

        csv_data = []
        frame_index = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                self.log("📁 End of video or failed to read a frame.")
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)
            hand_results = hands.process(image_rgb)

            row = [frame_index]

            if pose_results.pose_landmarks:
                for i in selected_pose_indices:
                    lm = pose_results.pose_landmarks.landmark[i]
                    row.append(lm.x)
                for i in selected_pose_indices:
                    lm = pose_results.pose_landmarks.landmark[i]
                    row.append(lm.y)
            else:
                row += [0] * len(selected_pose_indices) * 2

            handedness_dict = {}
            if hand_results.multi_handedness:
                for idx, hand_info in enumerate(hand_results.multi_handedness):
                    label = hand_info.classification[0].label.lower()
                    handedness_dict[label] = idx

            def get_hand_area(idx):
                if idx is None or idx >= len(hand_results.multi_hand_landmarks):
                    return 0
                landmarks = hand_results.multi_hand_landmarks[idx]
                points = []
                for lm in landmarks.landmark:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    points.append([px, py])
                if len(points) < 3:
                    return 0
                hull = cv2.convexHull(np.array(points))
                return int(cv2.contourArea(hull))

            left_area = get_hand_area(handedness_dict.get("left"))
            right_area = get_hand_area(handedness_dict.get("right"))
            row += [left_area, right_area]

            csv_data.append(row)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(image, f"Frame: {frame_index}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            out.write(image)
            frame_index += 1

            if frame_index % 10 == 0:
                self.log(f"⏱ Processed frame {frame_index}/{frame_count}")

        with open(output_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            writer.writerows(csv_data)

        cap.release()
        out.release()
        pose.close()
        hands.close()

        self.progress_label.config(text="✅ Completed!", fg="green")
        self.run_button.config(state="normal")
        self.log(f"✅ Done!\nCSV: {output_csv_path}\nVideo: {output_video_path}")