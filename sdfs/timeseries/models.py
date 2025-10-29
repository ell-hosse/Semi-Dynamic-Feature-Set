import torch
import torch.nn as nn

class SDFS(nn.Module):
    def __init__(
        self,
        static_input_size: int,
        dynamic_input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        output_size: int = 1,
        use_static_mlp: bool = False,
        static_mlp_hidden: int = 32,
    ):
        super().__init__()

        # optional tiny MLP to embed static features before repeating across time
        self.use_static_mlp = use_static_mlp
        if use_static_mlp:
            self.static_proj = nn.Sequential(
                nn.Linear(static_input_size, static_mlp_hidden),
                nn.ReLU(),
            )
            static_fused_size = static_mlp_hidden
        else:
            self.static_proj = nn.Identity()
            static_fused_size = static_input_size

        lstm_input_size = dynamic_input_size + static_fused_size

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        d = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * d, output_size)


    def forward(self, dynamic_seq: torch.Tensor, static_vec: torch.Tensor):
        B, T, _ = dynamic_seq.shape

        static_emb = self.static_proj(static_vec) # (B, S_static_emb)
        static_repeated = static_emb.unsqueeze(1).expand(B, T, -1) # (B, T, S_static_emb)

        fused = torch.cat([dynamic_seq, static_repeated], dim=-1) # (B, T, S_dynamic+S_static_emb)

        out, _ = self.lstm(fused) # (B, T, H * d)
        last = out[:, -1, :] # (B, H * d)
        y_hat = self.fc(self.dropout(last)) # (B, output_size)

        if y_hat.shape[-1] == 1:
            return y_hat.squeeze(-1) # (B,)
        return y_hat # (B, output_size)
