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
        use_attention: bool = False,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # PyTorch LSTM dropout only works with num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        if use_attention:
            # simple self-attention: learnable query vector
            self.attn_q = nn.Parameter(torch.randn(self.num_directions * hidden_dim))
        self.head = nn.Linear(self.num_directions * hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_dim * num_directions)
        if self.use_attention:
            # compute attention weights
            # out @ q → (batch, seq_len)
            scores = torch.matmul(out, self.attn_q)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len,1)
            # weighted sum
            context = (out * weights).sum(dim=1)  # (batch, hidden_dim * num_directions)
        else:
            # take last time-step
            context = out[:, -1, :]
        return self.head(context)