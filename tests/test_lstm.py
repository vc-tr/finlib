import torch
from src.models.lstm import PriceLSTM

def test_lstm_output_shape():
    batch, seq_len, feat = 8, 30, 10
    x = torch.randn(batch, seq_len, feat)
    model = PriceLSTM(input_dim=feat, hidden_dim=16, num_layers=1, use_attention=True)
    y = model(x)
    assert y.shape == (batch, 1)