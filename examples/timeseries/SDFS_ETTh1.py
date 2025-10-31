import os
import random
from typing import Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from matplotlib.backends.backend_svg import XMLWriter
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

from sdfs.timeseries.feature_expansion import sdfs
from examples.timeseries import ETTh1

CSV_PATH      = "examples/timeseries/ETTh1.csv"
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
    n_val = int(n * 0.10)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train+n_val].copy()
    test = df.iloc[n_train+n_val:].copy()
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


def run_example(dynamic_input_size):
    df = load_etth1(CSV_PATH)
    print("ETTh1 has been loaded successfully.")

    train_df, val_df, test_df = split_time_80_10_10(df)

    scaler = StandardScaler().fit(train_df[FEATURES].values)

    def build(split_df: pd.DataFrame):
        X = scaler.transform(split_df[FEATURES].values)
        y = split_df[TARGET_COL].values
        return make_windows(X, y, SEQ_LEN, HORIZON)

    Xw_train, yw_train = build(train_df)
    Xw_val, yw_val = build(val_df)
    Xw_test, yw_test = build(test_df)

    expanded_Xw_train , expanded_Xw_val, expanded_Xw_test = sdfs(Xw_train, Xw_val, Xw_test,
                                                                 yw_train, yw_val, yw_test,
                                                                 dynamic_input_size=dynamic_input_size)

    ETTh1.main(expanded_Xw_train.numpy(), expanded_Xw_val.numpy(), expanded_Xw_test.numpy(),
               yw_train, yw_val, yw_test, input_size=10)

