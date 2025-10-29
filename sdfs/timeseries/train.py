import torch
import torch.nn as nn
import torch.optim as optim
from sdfs.timeseries.early_stopping import EarlyStopping
from sdfs.timeseries.distances import find_closest_trend


def train(model, X_train, y_train, X_val, y_val, dynamic_features_list,
          num_epochs=50, learning_rate=1e-3, weight_decay=1e-5, patience=5):

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_values = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    print("Training Semi-Dynamic Feature Set (Time-Series Regression):")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for i in range(len(X_train)):
            static_input = X_train[i].unsqueeze(0)
            target = y_train[i].unsqueeze(0)

            # Prepare dynamic features
            dynamic_features = dynamic_features_list[i].unsqueeze(0).clone().detach().requires_grad_(True)

            optimizer.zero_grad()
            output = model(static_input, dynamic_features)

            # Regression loss
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Update dynamic features manually
            with torch.no_grad():
                dynamic_features -= dynamic_features.grad

            dynamic_features_list[i] = dynamic_features.detach().squeeze(0).clone()
            total_loss += loss.item()

        avg_loss = total_loss / len(X_train)
        loss_values.append(avg_loss)

        val_loss = validate(model, X_train, X_val, y_val, dynamic_features_list, criterion)

        print(f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Training completed.")
    return avg_loss, loss_values, dynamic_features_list


def validate(model, X_train, X_val, y_val, dynamic_features_list, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i in range(len(X_val)):
            static_input = X_val[i].unsqueeze(0)
            target = y_val[i].unsqueeze(0)

            dynamic_features = find_closest_trend(X_train, X_val[i], i, dynamic_features_list).unsqueeze(0)
            output = model(static_input, dynamic_features)

            loss = criterion(output, target)
            total_loss += loss.item()

    avg_loss = total_loss / len(X_val)
    return avg_loss
