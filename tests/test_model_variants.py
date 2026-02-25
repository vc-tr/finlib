import torch
import pytest
from src.experimental.models.lstm import PriceLSTM

@pytest.mark.parametrize("bidirectional, dropout, layer_norm, use_attention", [
    (False, 0.0, False, False),
    (True, 0.2, False, False),
    (False, 0.1, True, False),
    (True, 0.1, True, True),
])
def test_lstm_variants_shapes(bidirectional, dropout, layer_norm, use_attention):
    batch, seq_len, feat = 4, 20, 5
    model = PriceLSTM(
        input_dim=feat,
        hidden_dim=16,
        num_layers=2,
        bidirectional=bidirectional,
        dropout=dropout,
        layer_norm=layer_norm,
        use_attention=use_attention
    )
    x = torch.randn(batch, seq_len, feat)
    y = model(x)
    # Output must be (batch, 1)
    assert y.shape == (batch, 1)

    # Check parameter count roughly matches expectation:
    factor = 2 if bidirectional else 1
    expected_min = feat*16*factor  # very rough lower bound
    assert sum(p.numel() for p in model.parameters()) > expected_min