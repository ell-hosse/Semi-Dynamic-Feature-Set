import torch
import torch.nn as nn


class SDFS(nn.Module):
    def __init__(self, static_input_size: int, dynamic_input_size: int,
                 hidden_size: int=64, num_layers: int=1, dropout: float=0.1,
                 bidirectional: bool=False):

        super().__init__()
        self.lstm = nn.LSTM(
            input_size=static_input_size + dynamic_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        d = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * d, 1)

    def forward(self, static_input, dynamic_features):
        x = torch.cat((static_input, dynamic_features), dim=1)

        if x.dim() == 2:
            x = x.unsqueeze(0)

        out, _ = self.lstm(x)
        last = out[:, -1, :]
        yhat = self.fc(self.dropout(last))

        return yhat.squeeze(0)
