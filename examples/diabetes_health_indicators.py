import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from sdfs.feature_expansion import sdfs
from .classifier import Classifier, train, evaluate_model
import numpy as np


def load_dataset(test_size=0.2, validation_size=None, random_state=42):
    df = pd.read_csv(r'examples/diabetes_binary_health_indicators_BRFSS2015.csv')
    #df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

    df = df.rename(columns = {'Diabetes_012': 'Diabetes_binary'})
    df['Diabetes_binary'] = df['Diabetes_binary'].replace({2: 1})

    target = 'Diabetes_binary'

    def winsorize_iqr(df, col, k=1.5):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        # Replace values below/above the bounds with the boundary values
        return df[col].clip(lower=lower_bound, upper=upper_bound)

    df['BMI'] = winsorize_iqr(df, 'BMI')

    X = df.drop(target, axis=1).values
    y = df[target].values


    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if validation_size:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size,
                                                          random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_test, y_train, y_test


def run_example():
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(test_size=0.1, validation_size=0.1)

    print('Performance without SDFS:')
    classifier = Classifier(X_train.shape[1], 2)
    train(classifier, X_train, y_train, X_val, y_val, num_epochs=100)
    evaluate_model(classifier, X_test, y_test)
    print("-" * 50 + '\n\n')


    for dynamic_size in range(1, 11):
        print('Starting SDFS with DS size of :', dynamic_size)
        expanded_X_train, expanded_X_val, expanded_X_test = sdfs(X_train, X_val, X_test,
                                                                 y_train, y_val, y_test,
                                                                 num_classes=2,
                                                                 dynamic_input_size=4,
                                                                 init_method='PCA',
                                                                 distance_method='minkowski')

        print('Performance with SDFS:')
        classifier = Classifier(expanded_X_train.shape[1], 2)
        train(classifier, expanded_X_train, y_train, expanded_X_val, y_val, num_epochs=100)
        evaluate_model(classifier, expanded_X_test, y_test)
        print("-" * 50 + '\n\n')


