import numpy as np
from tslearn.metrics import dtw

def find_closest_trend(X_train, X_test, test_idx, dynamic_features_list, window_size=11):
    if window_size % 2 == 0: window_size += 1

    if test_idx <= window_size // 2:
        test_window = X_test[: window_size]
        corresponding_train_idx = test_idx

    elif (test_idx > window_size // 2 and
          len(X_test) - test_idx > window_size // 2):
        test_window = X_test[test_idx - window_size // 2: test_idx + window_size // 2 + 1]
        corresponding_train_idx = window_size // 2

    else:
        test_window = X_test[len(X_test) - window_size:]
        corresponding_train_idx = window_size - (len(X_test)-test_idx) - 1

    # Searching for the best matching window over training set
    best_train_idx, best_dist = None, float('inf')
    for i in range(len(X_train) - window_size + 1):
        train_window = X_train[i: i + window_size]

        dist = dtw(test_window, train_window)

        if dist < best_dist:
            best_train_idx, best_dist = i + corresponding_train_idx, dist

    return dynamic_features_list[best_train_idx]


if __name__ == '__main__':
    X_train = np.random.rand(20, 5)
    dynamic_features_list = np.random.rand(20, 2)

    X_test = X_train.copy()[5: 15]

    test_idx1 = 2; ws1 = 5; expected_train_idx1 = 7
    test_idx2 = 4; ws2 = 7; expected_train_idx2 = 9
    test_idx3 = 9; ws3 = 5; expected_train_idx3 = 13

    out1 = find_closest_trend(X_train, X_test, test_idx1, dynamic_features_list, ws1)[1]
    out2 = find_closest_trend(X_train, X_test, test_idx2, dynamic_features_list, ws2)[1]
    out3 = find_closest_trend(X_train, X_test, test_idx3, dynamic_features_list, ws3)[1]

    print(out1, expected_train_idx1)
    print(out2, expected_train_idx2)
    print(out3, expected_train_idx3)