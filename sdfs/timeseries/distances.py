import numpy as np
from fastdtw import fastdtw
from tslearn.metrics import dtw, lb_keogh, cdist_dtw


def find_closest_trend_normal(Xw_train, test_sequence, dynamic_features_list):
    best_train_seq_idx, best_dist = None, float('inf')

    for i, train_sequence in enumerate(Xw_train):
        dist = dtw(train_sequence, test_sequence)

        if dist < best_dist:
            best_dist = dist
            best_train_seq_idx = i

    print(best_train_seq_idx)

    return dynamic_features_list[best_train_seq_idx]


def find_closest_trend_fastdtw(Xw_train, test_sequence, dynamic_features_list):
    q = np.asarray(test_sequence, dtype=float)
    best_idx, best_dist = None, float('inf')
    for i, x in enumerate(Xw_train):
        dist, _ = fastdtw(np.asarray(x, float), q)
        if dist < best_dist:
            best_dist, best_idx = dist, i

    print(best_idx)

    return dynamic_features_list[best_idx]


def find_closest_trend_cdist_dtw(Xw_train, test_sequence, dynamic_features_list):
    X = np.asarray(Xw_train, dtype=float)
    q = np.asarray(test_sequence, dtype=float)[None, ...]

    D = cdist_dtw(
        X, q,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=5,
        n_jobs=-1
    )
    best_idx = int(np.argmin(D[:, 0]))

    print(best_idx)

    return dynamic_features_list[best_idx]


def _znorm_along_time(X, eps=1e-8):
    mean = X.mean(axis=-2, keepdims=True)
    std  = X.std(axis=-2, keepdims=True)
    return (X - mean) / (std + eps)

def find_closest_trend_znorm(Xw_train, test_sequence, dynamic_features_list):
    N, T, F = Xw_train.shape
    Xz = _znorm_along_time(Xw_train.astype(np.float32))
    yz = _znorm_along_time(test_sequence.astype(np.float32)[None, ...])[0]

    X = Xz.reshape(N, -1)
    y = yz.reshape(-1)

    X_norm = np.linalg.norm(X, axis=1) + 1e-8
    y_norm = np.linalg.norm(y) + 1e-8
    sims = (X @ y) / (X_norm * y_norm)
    idx = int(np.argmax(sims))

    return dynamic_features_list[idx]


def _znorm_along_time(X, eps=1e-8):
    mean = X.mean(axis=-2, keepdims=True)
    std  = X.std(axis=-2, keepdims=True)
    return (X - mean) / (std + eps)

def find_closest_trend(Xw_train, test_sequence, dynamic_features_list, eps=1e-8):
    # shape-only similarity (z-norm + cosine)
    N, T, F = Xw_train.shape
    Xz = _znorm_along_time(Xw_train.astype(np.float32))
    yz = _znorm_along_time(test_sequence.astype(np.float32)[None, ...])[0]

    X = Xz.reshape(N, -1)
    y = yz.reshape(-1)

    X_norm = np.linalg.norm(X, axis=1) + eps
    y_norm = np.linalg.norm(y) + eps
    sims = (X @ y) / (X_norm * y_norm)
    idx = int(np.argmax(sims))

    # compute scale factor between raw sequences
    Xi = Xw_train[idx].astype(np.float64)
    Y  = test_sequence.astype(np.float64)

    # center each by its feature-wise mean across time
    mu_x = Xi.mean(axis=0, keepdims=True)
    mu_y = Y.mean(axis=0,  keepdims=True)
    Xc = Xi - mu_x
    Yc = Y  - mu_y

    # global scale (Frobenius norm ratio)
    scale = float(np.linalg.norm(Yc) / (np.linalg.norm(Xc) + eps))

    # scale the selected dynamic features
    dyn = np.asarray(dynamic_features_list[idx], dtype=np.float64)
    dyn_scaled = dyn * scale

    #print(idx, scale)

    # return scaled dynamic features
    return dyn_scaled


if __name__ == '__main__':
    from time import time
    # Testing the correctness and time duration of different approaches

    Xw_train = np.random.random((14000, 48, 6))
    dynamic_features_list = np.random.random((140000, 48, 3))
    test_sequence = Xw_train[12000] * 3

    test_sequence[:, 3] *= 12
    test_sequence[:, 5] *= 4

    start = time()
    find_closest_trend(Xw_train, test_sequence, dynamic_features_list)
    end = time()
    print('Z-Norm (scaled): ', end - start)

    start = time()
    find_closest_trend_znorm(Xw_train, test_sequence, dynamic_features_list)
    end = time()
    print('Z-Norm: ', end - start)

    start = time()
    find_closest_trend_normal(Xw_train, test_sequence, dynamic_features_list)
    end = time()
    print('DTW: ', end - start)

    start = time()
    find_closest_trend_fastdtw(Xw_train, test_sequence, dynamic_features_list)
    end = time()
    print('Fast DTW: ', end - start)

    start = time()
    find_closest_trend_cdist_dtw(Xw_train, test_sequence, dynamic_features_list)
    end = time()
    print('CDist DTW: ', end - start)
