"""
Machine-learning signal models for the forecast platform.

This subpackage adds *learned* alpha signals on top of the rule-based strategy
catalog. The defining constraint is the same one the rest of the platform
enforces: **no lookahead**. Models are retrained walk-forward on an expanding
window, and a training row is dropped whenever its forward-looking label could
overlap the point being predicted (see ``walkforward.walk_forward_signal``).

Public API:
    make_features      -- causal feature matrix from a price series
    make_labels        -- forward-return direction labels (training target only)
    walk_forward_signal -- model-agnostic causal predictor -> signal in {-1,0,1}
    WalkForwardConfig  -- knobs for retrain cadence / horizon / flat band
    SklearnDirectionModel -- logistic / gradient-boosting direction model
    LSTMDirectionModel    -- small PyTorch sequence model (optional dependency)
"""

from .features import make_features, make_labels, FEATURE_NAMES
from .walkforward import walk_forward_signal, WalkForwardConfig, DirectionModel
from .sklearn_model import SklearnDirectionModel, logistic_factory, gradient_boost_factory

__all__ = [
    "make_features",
    "make_labels",
    "FEATURE_NAMES",
    "walk_forward_signal",
    "WalkForwardConfig",
    "DirectionModel",
    "SklearnDirectionModel",
    "logistic_factory",
    "gradient_boost_factory",
]
