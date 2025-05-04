from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression


def apply_polynomial(X_train, X_val, X_test, y_train, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler()
    X_train_poly = scaler.fit_transform(poly.fit_transform(X_train))
    X_val_poly = scaler.transform(poly.transform(X_val))
    X_test_poly = scaler.transform(poly.transform(X_test))
    clf = LogisticRegression(max_iter=100)
    clf.fit(X_train_poly, y_train)
    return X_train_poly, X_val_poly, X_test_poly