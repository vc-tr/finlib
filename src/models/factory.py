from typing import Type
from src.models.lstm import PriceLSTM
from src.models.gru import PriceGRU
from src.models.bilstm import PriceBiLSTM

_MODEL_MAP = {
    "lstm": PriceLSTM,
    "gru": PriceGRU,
    "bilstm": PriceBiLSTM,
}

def get_model_class(name: str) -> Type:
    return _MODEL_MAP[name]