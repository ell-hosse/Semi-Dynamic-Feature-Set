import os
import random
from typing import Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_PATH      = "ETTh1.csv"
FEATURES      = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET_COL    = "OT"

SEQ_LEN       = 48   # use past 48 hours
HORIZON       = 1    # predict OT at t+1 hour
BATCH_SIZE    = 128
LR            = 1e-3
EPOCHS        = 50
PATIENCE      = 7

HIDDEN_SIZE   = 64
NUM_LAYERS    = 1
BIDIRECTIONAL = False
DROPOUT       = 0.1

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)


def load_etth1(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at '{csv_path}'.")
    df = pd.read_csv(csv_path)
    expected = {"date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def split_time_80_10_10(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)
    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train:n_train+n_val].copy()
    test  = df.iloc[n_train+n_val:].copy()
    return train, val, test


def make_windows(X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int):
    T = len(y)
    last_start = T - (seq_len + horizon) + 1
    xs, ys = [], []
    for start in range(last_start):
        end = start + seq_len
        target_idx = end + horizon - 1
        xs.append(X[start:end, :])
        ys.append(y[target_idx])
    return np.stack(xs), np.asarray(ys)


class WindowDataset(Dataset):
    def __init__(self, Xw: np.ndarray, y: np.ndarray):
        self.X = Xw.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        d = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * d, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        yhat = self.fc(self.dropout(last))
        train = self.training
        return yhat.squeeze(-1)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def predict(model, loader, device):
    model.eval()
    yp, yt = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            yp.append(pred)
            yt.append(yb.numpy())
    return np.concatenate(yp), np.concatenate(yt)


def regression_report(y_true, y_pred, name=""):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{name}MAE : {mae:.4f}")
    print(f"{name}RMSE: {rmse:.4f}")
    print(f"{name}R2  : {r2:.4f}")
    return mae, rmse, r2


def main():
    print("Loading ETTh1...")
    df = load_etth1(CSV_PATH)

    # Chronological 80/10/10 split
    train_df, val_df, test_df = split_time_80_10_10(df)

    scaler = StandardScaler().fit(train_df[FEATURES].values)

    def build(split_df: pd.DataFrame):
        X = scaler.transform(split_df[FEATURES].values)
        y = split_df[TARGET_COL].values
        return make_windows(X, y, SEQ_LEN, HORIZON)

    Xw_tr, yw_tr = build(train_df)
    Xw_va, yw_va = build(val_df)
    Xw_te, yw_te = build(test_df)

    dl_tr = DataLoader(WindowDataset(Xw_tr, yw_tr), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=False)
    dl_va = DataLoader(WindowDataset(Xw_va, yw_va), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    dl_te = DataLoader(WindowDataset(Xw_te, yw_te), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)

    model = LSTMRegressor(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT
    ).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train with early stopping on val MAE
    best_va, best_state, patience = np.inf, None, PATIENCE
    print("Training...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, dl_tr, criterion, optimizer, DEVICE)
        yv_pred, yv_true = predict(model, dl_va, DEVICE)
        va_mae = mean_absolute_error(yv_true, yv_pred)
        print(f"[Epoch {epoch:02d}] train_MAE={tr_loss:.4f} | val_MAE={va_mae:.4f}")
        if va_mae < best_va:
            best_va, best_state, patience = va_mae, {k: v.cpu().clone() for k, v in model.state_dict().items()}, PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    print("\n=== Validation ===")
    yv_pred, yv_true = predict(model, dl_va, DEVICE)
    regression_report(yv_true, yv_pred, name="Val ")

    print("\n=== Test ===")
    yt_pred, yt_true = predict(model, dl_te, DEVICE)
    regression_report(yt_true, yt_pred, name="Test ")

    # Naive baseline: predict next OT = last OT in the input window
    print("\n=== Naive baseline (last OT) on Test ===")
    # Last OT from each window is column index of OT in FEATURES
    ot_idx = FEATURES.index("OT")
    naive_pred = Xw_te[:, -1, ot_idx]
    regression_report(yw_te, naive_pred, name="Naive ")

if __name__ == "__main__":
    main()
