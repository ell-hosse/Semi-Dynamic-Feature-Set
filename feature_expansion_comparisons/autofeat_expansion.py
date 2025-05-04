from autofeat import AutoFeatClassifier
from sklearn.linear_model import LogisticRegression


def apply_autofeat(X_train, X_val, X_test, y_train):
    model = AutoFeatClassifier(verbose=0, feateng_steps=2)
    model.fit(X_train, y_train)
    X_train_new = model.transform(X_train)
    X_val_new = model.transform(X_val)
    X_test_new = model.transform(X_test)
    clf = LogisticRegression(max_iter=10)
    clf.fit(X_train_new, y_train)
    return X_train_new, X_val_new, X_test_new