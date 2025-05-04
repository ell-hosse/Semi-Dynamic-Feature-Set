import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def apply_pytorch_autoencoder(X_train, X_val, X_test, y_train, encoding_dim=15, epochs=50, batch_size=32):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    train_loader = data.DataLoader(X_train_tensor, batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_dim=X_train.shape[1], encoding_dim=encoding_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_train_encoded = model.encoder(X_train_tensor).numpy()
        X_val_encoded = model.encoder(torch.tensor(X_val, dtype=torch.float32)).numpy()
        X_test_encoded = model.encoder(torch.tensor(X_test, dtype=torch.float32)).numpy()

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train_encoded, y_train)

    return X_train_encoded, X_val_encoded, X_test_encoded

