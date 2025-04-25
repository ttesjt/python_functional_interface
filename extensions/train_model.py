import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

LABELS = [
    "Idle", "Guard", "Punch_Left", "Punch_Right",
    "Cross_Left", "Cross_Right", "Uppercut_Left", "Uppercut_Right"
]

# ─────────────────────────────────────────────
def normalize_landmark_coords(df: pd.DataFrame) -> pd.DataFrame:
    df_norm = df.copy()
    left_x = df["pose_11_x"]
    left_y = df["pose_11_y"]
    right_x = df["pose_12_x"]
    right_y = df["pose_12_y"]
    mid_x = (left_x + right_x) / 2
    mid_y = (left_y + right_y) / 2
    unit_len = np.sqrt((left_x - right_x)**2 + (left_y - right_y)**2)
    unit_len = unit_len.replace(0, np.nan)
    for col in df.columns:
        if col.endswith("_x"):
            df_norm[col] = (df[col] - mid_x) / unit_len
        elif col.endswith("_y"):
            df_norm[col] = (df[col] - mid_y) / unit_len
    return df_norm

def extract_useful_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["state"].notna() & (df["state"] != "")].copy()
    df_out = pd.DataFrame()
    df_out["frame"] = df["frame"]
    df_out["state"] = df["state"]
    body_indices = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]
    for idx in body_indices:
        df_out[f"pose_{idx}_x"] = df[f"pose_{idx}_x"]
        df_out[f"pose_{idx}_y"] = df[f"pose_{idx}_y"]
    hand_indices = [0, 5, 6, 17, 18]
    for hand in ['L', 'R']:
        for idx in hand_indices:
            df_out[f"hand{hand}_{idx}_x"] = df[f"hand{hand}_{idx}_x"]
            df_out[f"hand{hand}_{idx}_y"] = df[f"hand{hand}_{idx}_y"]
    def compute_box_area(points):
        x = points[:, 0]
        y = points[:, 1]
        return (x.max() - x.min()) * (y.max() - y.min())
    def extract_area_set(row, hand_prefix, indices):
        points = np.array([
            [row[f"{hand_prefix}_{i}_x"], row[f"{hand_prefix}_{i}_y"]]
            for i in indices
        ])
        return compute_box_area(points)
    box1_indices = [5, 6, 17, 18]
    box2_indices = [0, 5, 17]
    left_box1, left_box2, right_box1, right_box2 = [], [], [], []
    for _, row in df.iterrows():
        left_box1.append(extract_area_set(row, "handL", box1_indices))
        left_box2.append(extract_area_set(row, "handL", box2_indices))
        right_box1.append(extract_area_set(row, "handR", box1_indices))
        right_box2.append(extract_area_set(row, "handR", box2_indices))
    df_out["left_box1"] = left_box1
    df_out["left_box2"] = left_box2
    df_out["right_box1"] = right_box1
    df_out["right_box2"] = right_box2
    df_out = normalize_landmark_coords(df_out)
    return df_out

def prepare_data(df, label_encoder, window_size):
    X_all = df.drop(columns=["frame", "state"]).values
    y_all = df["state"].values
    X_seq, y_seq = [], []
    for i in range(len(df) - window_size):
        label = y_all[i + window_size - 1]
        if label not in LABELS:
            continue
        x_window = X_all[i:i + window_size]
        y_label = label_encoder.transform([label])[0]
        X_seq.append(x_window)
        y_seq.append(y_label)
    X_seq = np.array(X_seq).astype(np.float32)
    y_seq = np.array(y_seq).astype(np.int64)
    return torch.tensor(X_seq), torch.tensor(y_seq)

class TemporalCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

def compute_action_met(true_labels, pred_labels):
    met_flags = np.zeros(len(true_labels), dtype=bool)
    current_chunk = []
    for i in range(len(true_labels)):
        label = LABELS[true_labels[i]]
        if label in ["Idle", "Guard"]:
            if current_chunk:
                start = current_chunk[0]
                end = current_chunk[-1]
                context = list(range(max(0, start - 2), min(len(true_labels), end + 3)))
                target_label = true_labels[end]
                if any(pred_labels[j] == target_label for j in context):
                    for j in current_chunk:
                        met_flags[j] = True
                current_chunk = []
        else:
            current_chunk.append(i)
    if current_chunk:
        start = current_chunk[0]
        end = current_chunk[-1]
        context = list(range(max(0, start - 2), min(len(true_labels), end + 3)))
        target_label = true_labels[end]
        if any(pred_labels[j] == target_label for j in context):
            for j in current_chunk:
                met_flags[j] = True
    return met_flags

def main(args):
    window_size = args.window or 6
    batch_size = args.batch or 32
    epochs = args.epochs or 30
    lr = args.lr or 0.001

    train_df = extract_useful_features(pd.read_csv(args.train))
    test_df = extract_useful_features(pd.read_csv(args.test))

    le = LabelEncoder()
    le.classes_ = np.array(LABELS)

    X_train, y_train = prepare_data(train_df, le, window_size)
    X_test, y_test = prepare_data(test_df, le, window_size)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalCNN(X_train.shape[2], len(LABELS)).to(device)

    if args.load:
        if os.path.exists(args.load):
            model.load_state_dict(torch.load(args.load, map_location=device))
            print(f" Loaded model from: {args.load}")
        else:
            print(f" Could not find model at {args.load}. Training from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []
    print(f"Training started on {device}...")

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            predicted = pred.argmax(dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2%}")

    # ───── Evaluation ─────────────────────────────
    model.eval()
    all_true_labels, all_predicted_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            predicted = pred.argmax(dim=1)
            for t, p in zip(y_batch.cpu(), predicted.cpu()):
                if LABELS[t] == "Idle" and LABELS[p] == "Idle":
                    continue  # skip Idle-Idle match
                total += 1
                if t == p:
                    correct += 1
            all_true_labels.extend(y_batch.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

    print(f"\n Accuracy (excluding Idle-Idle matches): {correct / total:.2%}")

    # Save model
    os.makedirs("data/model_saves", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"data/model_saves/model_{timestamp}.pth"
    torch.save(model.state_dict(), save_path)
    print(f" Model saved to: {save_path}")

    # Training charts
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Prediction scatter
    plt.figure(figsize=(12, 5))
    plt.scatter(range(len(all_true_labels)), all_true_labels, label="True", alpha=0.5)
    plt.scatter(range(len(all_predicted_labels)), all_predicted_labels, label="Predicted", alpha=0.5, marker='x', c='red')
    plt.yticks(ticks=list(range(len(LABELS))), labels=LABELS)
    plt.xlabel("Frame Index")
    plt.ylabel("Label")
    plt.title("Expected vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Action Met Visualization
    action_met_flags = compute_action_met(all_true_labels, all_predicted_labels)
    plt.figure(figsize=(10, 1))
    plt.title("Action Met Timeline (Green=Met, Red=Missed)")
    for i, flag in enumerate(action_met_flags):
        color = "green" if flag else "red"
        plt.axvline(i, color=color, alpha=0.8)
    plt.yticks([])
    plt.xlabel("Frame Index")
    plt.tight_layout()
    plt.show()

    print(f"\n Actions met: {np.sum(action_met_flags)} / {len(action_met_flags)}")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training CSV file")
    parser.add_argument("--test", required=True, help="Path to testing CSV file")
    parser.add_argument("--load", type=str, default="", help="Path to a .pth model file to resume training")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--window", type=int, default=0)
    args = parser.parse_args()
    main(args)