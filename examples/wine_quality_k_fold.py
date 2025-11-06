import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sdfs.feature_expansion import sdfs
from .classifier import Classifier, train

def _to_tensor(x, y):
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def _eval_logits(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y.cpu().numpy(), y_pred)
    f1 = f1_score(y.cpu().numpy(), y_pred)
    try:
        auc = roc_auc_score(y.cpu().numpy(), probs[:, 1])
    except:
        auc = np.nan
    return acc, f1, auc

def load_wine_quality_full(path='examples/winequalityN.csv'):
    df = pd.read_csv(path)
    for col in df.columns[df.isnull().any()]:
        df[col] = df[col].fillna(df[col].mean())
    X = df.iloc[:, 1:-1].values
    y = (df.iloc[:, -1] > 6).astype(int).values
    return X, y

def run_cv_80_20(n_splits=5, val_size=0.2, epochs=100, dynamic_input_size=4, random_state=42):
    X_all, y_all = load_wine_quality_full()
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=random_state)

    base_scores, sdfs_scores = [], []

    split_id = 0
    for train_idx, val_idx in splitter.split(X_all, y_all):
        split_id += 1
        X_tr_raw, X_val_raw = X_all[train_idx], X_all[val_idx]
        y_tr_raw, y_val_raw = y_all[train_idx], y_all[val_idx]

        scaler = StandardScaler().fit(X_tr_raw)
        X_tr_s = scaler.transform(X_tr_raw)
        X_val_s = scaler.transform(X_val_raw)

        X_tr, y_tr = _to_tensor(X_tr_s, y_tr_raw)
        X_val, y_val = _to_tensor(X_val_s, y_val_raw)

        clf = Classifier(X_tr.shape[1], 2)
        train(clf, X_tr, y_tr, X_val, y_val, num_epochs=epochs)
        base_scores.append(_eval_logits(clf, X_val, y_val))

        X_tr_exp, X_val_exp, X_val_as_test_exp = sdfs(
            X_tr, X_val, X_val, y_tr, y_val, y_val,
            num_classes=2, dynamic_input_size=dynamic_input_size,
            init_method='PCA', distance_method='minkowski'
        )

        clf_sdfs = Classifier(X_tr_exp.shape[1], 2)
        train(clf_sdfs, X_tr_exp, y_tr, X_val_exp, y_val, num_epochs=epochs)
        sdfs_scores.append(_eval_logits(clf_sdfs, X_val_as_test_exp, y_val))

        b = base_scores[-1]; s = sdfs_scores[-1]
        print(f'split {split_id} | Baseline acc={b[0]:.3f} f1={b[1]:.3f} auc={b[2]:.3f} | SDFS acc={s[0]:.3f} f1={s[1]:.3f} auc={s[2]:.3f}')

    base_arr = np.array(base_scores, float)
    sdfs_arr = np.array(sdfs_scores, float)

    def ms(a): return a.mean(axis=0), a.std(axis=0)

    b_mean, b_std = ms(base_arr)
    s_mean, s_std = ms(sdfs_arr)

    print('-'*50)
    print(f'Baseline mean±std | acc={b_mean[0]:.4f}±{b_std[0]:.4f} f1={b_mean[1]:.4f}±{b_std[1]:.4f} auc={b_mean[2]:.4f}±{b_std[2]:.4f}')
    print(f'SDFS mean±std | acc={s_mean[0]:.4f}±{s_std[0]:.4f} f1={s_mean[1]:.4f}±{s_std[1]:.4f} auc={s_mean[2]:.4f}±{s_std[2]:.4f}')

