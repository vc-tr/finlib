from typing import Type
from src.models.lstm import PriceLSTM
from src.models.gru import PriceGRU
from src.models.bilstm import PriceBiLSTM
from src.models.transformer import PriceTransformer
from src.models.tcn import PriceTCN

_MODEL_MAP = {
    "lstm": PriceLSTM,
    "gru": PriceGRU,
    "bilstm": PriceBiLSTM,
    "transformer": PriceTransformer,
    "tcn": PriceTCN,
}


def get_model_class(name: str) -> Type:
    if name not in _MODEL_MAP:
        raise ValueError(
            f"Unknown model: {name}. Choose from: {list(_MODEL_MAP.keys())}"
        )
    return _MODEL_MAP[name]


def create_model(
    name: str,
    input_dim: int = 1,
    hidden_dim: int = 64,
    **kwargs,
):
    """Create model with sensible defaults for each type."""
    cls = get_model_class(name)
    if name in ("lstm", "gru", "bilstm"):
        return cls(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)
    if name == "transformer":
        return cls(input_dim=input_dim, d_model=hidden_dim, **kwargs)
    if name == "tcn":
        return cls(input_dim=input_dim, n_channels=hidden_dim, **kwargs)
    return cls(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)