"""DL models moved to src.experimental.models. Use: from src.experimental.models import PriceLSTM, create_model."""

def __getattr__(name: str):
    raise ImportError(
        "DL models moved to src.experimental.models. "
        "Use: from src.experimental.models import PriceLSTM, create_model"
    )
