import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sdfs.feature_expansion import sdfs
from .classifier import Classifier, train, evaluate_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split


def load_wine_quality_data(test_size=0.2, validation_size=None, random_state=42):
    df = pd.read_csv(r'examples/winequalityN.csv')

    for col in df.columns[df.isnull().any()]:
        df[col] = df[col].fillna(df[col].mean())

    X = df.iloc[:, 1: -1].values
    y = (df.iloc[:, -1] > 6).astype(int).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if validation_size:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size,
                                                          random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_test, y_train, y_test

def _rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _eval_to_tuple(metrics_dict):
    acc = metrics_dict['accuracy'][0]
    f1 = metrics_dict['f1_score'][0]
    auc = metrics_dict['roc_auc'][0] if metrics_dict['roc_auc'][0] is not None else np.nan
    return acc, f1, auc

def _evaluate_logits(model, X, y):
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

def run_once(seed, dynamic_input_size=4):
    _rng(seed)
    X_train, X_val, X_test, y_train, y_val, y_test = load_wine_quality_data(test_size=0.1, validation_size=0.1, random_state=seed)

    clf = Classifier(X_train.shape[1], 2)
    train(clf, X_train, y_train, X_val, y_val, num_epochs=100)
    base_metrics = _evaluate_logits(clf, X_test, y_test)

    expanded_X_train, expanded_X_val, expanded_X_test = sdfs(
        X_train, X_val, X_test, y_train, y_val, y_test,
        num_classes=2, dynamic_input_size=dynamic_input_size, init_method='PCA', distance_method='minkowski'
    )
    clf_sdfs = Classifier(expanded_X_train.shape[1], 2)
    train(clf_sdfs, expanded_X_train, y_train, expanded_X_val, y_val, num_epochs=100)
    sdfs_eval = _evaluate_logits(clf_sdfs, expanded_X_test, y_test)

    return base_metrics, sdfs_eval

def summarize(arr):
    arr = np.asarray(arr, dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)

def run_random_splits(seeds=(1,2,3,4,5), dynamic_input_size=4):
    base_scores = []
    sdfs_scores = []
    for s in seeds:
        base, sdfs_res = run_once(s, dynamic_input_size=dynamic_input_size)
        base_scores.append(list(base))
        sdfs_scores.append(list(sdfs_res))
        print(f'seed {s} | Baseline acc={base[0]:.3f} f1={base[1]:.3f} auc={base[2]:.3f} | SDFS acc={sdfs_res[0]:.3f} f1={sdfs_res[1]:.3f} auc={sdfs_res[2]:.3f}')
    base_mean, base_std = summarize(base_scores)
    sdfs_mean, sdfs_std = summarize(sdfs_scores)
    print('-'*50)
    print(f'Baseline mean±std | acc={base_mean[0]:.4f}±{base_std[0]:.4f} f1={base_mean[1]:.4f}±{base_std[1]:.4f} auc={base_mean[2]:.4f}±{base_std[2]:.4f}')
    print(f'SDFS mean±std | acc={sdfs_mean[0]:.4f}±{sdfs_std[0]:.4f} f1={sdfs_mean[1]:.4f}±{sdfs_std[1]:.4f} auc={sdfs_mean[2]:.4f}±{sdfs_std[2]:.4f}')

