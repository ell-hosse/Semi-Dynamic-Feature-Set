import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

XLSX_PATH   = "AirQualityUCI.xlsx"

FEATURES    = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "T", "RH", "AH", "NO2(GT)"]
TARGET_COL  = "NO2(GT)"

SEQ_LEN     = 48
HORIZON     = 1
BATCH_SIZE  = 256
LR          = 1e-3
EPOCHS      = 100
PATIENCE    = 7
HIDDEN_SIZE = 64

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_airquality(xlsx_path: str) -> pd.DataFrame:
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"File not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path)

    missing = [c for c in set(FEATURES + [TARGET_COL]) if c not in df.columns]
    if missing:
        have = list(df.columns)
        raise ValueError(f"Missing columns: {missing}\nFirst available columns: {have[:20]}")

    if "Date" in df.columns and "Time" in df.columns:
        df["dt"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
            dayfirst=True
        )
        df = df.sort_values("dt").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    used_cols = list(set(FEATURES + [TARGET_COL]))
    clean = df.copy()
    for c in used_cols:
        clean = clean[clean[c] != -200]

    clean = clean[used_cols].dropna().reset_index(drop=True)

    if len(clean) < (SEQ_LEN + HORIZON + 10):
        raise ValueError(
            f"Not enough clean rows ({len(clean)}) for SEQ_LEN={SEQ_LEN} & HORIZON={HORIZON}."
        )
    return clean


def split_time_80_10_10(df: pd.DataFrame):
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
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


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
            pred = model(xb.to(device)).cpu().numpy()
            yp.append(pred); yt.append(yb.numpy())
    return np.concatenate(yp), np.concatenate(yt)


def regression_report(y_true, y_pred, name=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2   = r2_score(y_true, y_pred)
    print(f"{name}MAE : {mae:.4f}")
    print(f"{name}RMSE: {rmse:.4f}")
    print(f"{name}R2  : {r2:.4f}")
    return mae, rmse, r2


def main(Xw_tr, Xw_va, Xw_te, yw_tr, yw_va, yw_te, input_size):
    dl_tr = DataLoader(WindowDataset(Xw_tr, yw_tr), batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(WindowDataset(Xw_va, yw_va), batch_size=BATCH_SIZE, shuffle=False)
    dl_te = DataLoader(WindowDataset(Xw_te, yw_te), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(input_size=input_size, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_va, best_state, patience = np.inf, None, PATIENCE
    print("Training...")
    for epoch in range(1, EPOCHS + 1):
        tr_mae = train_one_epoch(model, dl_tr, criterion, optimizer, DEVICE)
        yv_pred, yv_true = predict(model, dl_va, DEVICE)
        va_mae = mean_absolute_error(yv_true, yv_pred)
        print(f"[Epoch {epoch:02d}] train_MAE={tr_mae:.4f} | val_MAE={va_mae:.4f}")
        if va_mae < best_va:
            best_va, patience = va_mae, PATIENCE
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping."); break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n=== Validation ===")
    yv_pred, yv_true = predict(model, dl_va, DEVICE)
    regression_report(yv_true, yv_pred, name="Val ")

    print("\n=== Test ===")
    yt_pred, yt_true = predict(model, dl_te, DEVICE)
    regression_report(yt_true, yt_pred, name="Test ")

    print("\n=== Naive baseline (last NO2(GT)) on Test ===")
    no2_idx = FEATURES.index(TARGET_COL)
    naive_pred = Xw_te[:, -1, no2_idx]
    regression_report(yw_te, naive_pred, name="Naive ")


if __name__ == "__main__":
    set_seed(SEED)
    print("Loading AirQualityUCI (full, cleaned)...")
    df = load_airquality(XLSX_PATH)

    train_df, val_df, test_df = split_time_80_10_10(df)

    scaler = StandardScaler().fit(train_df[FEATURES].values)

    def build(split_df: pd.DataFrame):
        X = scaler.transform(split_df[FEATURES].values)
        y = split_df[TARGET_COL].values
        return make_windows(X, y, SEQ_LEN, HORIZON)

    Xw_train, yw_train = build(train_df)
    Xw_val, yw_val = build(val_df)
    Xw_test, yw_test = build(test_df)
    main(Xw_train, Xw_val, Xw_test, yw_train, yw_val, yw_test, input_size=len(FEATURES))
