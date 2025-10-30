import torch
import torch.nn as nn

from sdfs.timeseries.early_stopping import EarlyStopping
from sdfs.timeseries.distances import find_closest_trend


def train(model, Xw_train, yw_train, Xw_val, yw_val, dynamic_features_list,
          num_epochs=50, learning_rate=1e-3, weight_decay=1e-5, patience=5):

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_values = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    print(">> Training Semi-Dynamic Feature Set (Time-Series Regression):")
    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0

        for i in range(len(Xw_train)):
            static_input = torch.tensor(Xw_train[i], dtype=torch.float32)
            target = torch.tensor(yw_train[i]).unsqueeze(0)

            # Prepare dynamic features
            dynamic_features = dynamic_features_list[i].clone().detach().requires_grad_(True)

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

        avg_loss = total_loss / len(Xw_train)
        loss_values.append(avg_loss)

        val_loss = validate(model, Xw_train, Xw_val, yw_val, dynamic_features_list, criterion)

        print(f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Training has been completed successfully.")
    return avg_loss, loss_values, dynamic_features_list


def validate(model, Xw_train, Xw_val, yw_val, dynamic_features_list, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i in range(len(Xw_val)):
            static_input = torch.tensor(Xw_val[i], dtype=torch.float32)
            target = torch.tensor(yw_val[i]).unsqueeze(0)

            dynamic_features = find_closest_trend(Xw_train, Xw_val[i], dynamic_features_list)
            output = model(static_input, dynamic_features)

            loss = criterion(output, target)
            total_loss += loss.item()

    avg_loss = total_loss / len(Xw_val)
    return avg_loss
