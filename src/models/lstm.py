import torch
import torch.nn as nn

class PriceLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.1,
        layer_norm: bool = False,
        use_attention: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0.0,
            bidirectional=bidirectional,
        )

        # Optional layer-norm after LSTM outputs
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self.num_directions * hidden_dim)
        else:
            self.layer_norm = None

        # Optional attention: learnable query
        self.use_attention = use_attention
        if use_attention:
            self.attn_q = nn.Parameter(torch.randn(self.num_directions * hidden_dim))

        # Final head
        self.head = nn.Linear(self.num_directions * hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, 1)
        """
        out, _ = self.lstm(x)  # → (batch, seq_len, hidden_dim * num_directions)
        if self.layer_norm:
            # apply layernorm per time step
            out = self.layer_norm(out)

        if self.use_attention:
            # out @ q → (batch, seq_len)
            scores = torch.matmul(out, self.attn_q)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
            context = (out * weights).sum(dim=1)                 # (batch, hidden_dim * num_directions)
        else:
            # take the last time step
            context = out[:, -1, :]

        return self.head(context)