from tslearn.metrics import dtw

def find_closest_trend(Xw_train, test_sequence, dynamic_features_list):
    best_train_seq_idx, best_dist = None, float('inf')

    for i, train_sequence in enumerate(Xw_train):
        dist = dtw(train_sequence, test_sequence)

        if dist < best_dist:
            best_dist = dist
            best_train_seq_idx = i

    return dynamic_features_list[best_train_seq_idx]