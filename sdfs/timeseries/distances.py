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

    return dynamic_features_list[best_train_seq_idx]


def find_closest_trend_lbkeogh(Xw_train, test_sequence, dynamic_features_list,
                               K=200, radius=5):
    q = np.asarray(test_sequence, dtype=float)
    lbs = np.array([lb_keogh(q, np.asarray(x, float), radius=radius) for x in Xw_train])
    cand_idx = np.argpartition(lbs, K)[:K]

    best_idx, best_dist = None, float('inf')
    for i in cand_idx:
        dist = dtw(np.asarray(Xw_train[i], float), q)
        if dist < best_dist:
            best_dist, best_idx = dist, i
    return dynamic_features_list[best_idx]


def find_closest_trend_fastdtw(Xw_train, test_sequence, dynamic_features_list):
    q = np.asarray(test_sequence, dtype=float)
    best_idx, best_dist = None, float('inf')
    for i, x in enumerate(Xw_train):
        dist, _ = fastdtw(np.asarray(x, float), q)
        if dist < best_dist:
            best_dist, best_idx = dist, i
    return dynamic_features_list[best_idx]


def find_closest_trend(Xw_train, test_sequence, dynamic_features_list):
    X = np.asarray(Xw_train, dtype=float)
    q = np.asarray(test_sequence, dtype=float)[None, ...]

    D = cdist_dtw(
        X, q,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=5,
        n_jobs=-1
    )
    best_idx = int(np.argmin(D[:, 0]))
    return dynamic_features_list[best_idx]

