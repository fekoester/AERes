"""AERes: attention-enhanced reservoir computing in PyTorch."""

__version__ = "0.2.0"

from .reservoir import Reservoir
from .dynamicalSystems import DynamicalSystemSimulator, LorenzSimulator
from .readout import RidgeTrainer
from .metrics import mse, nrmse, normalized_error, valid_prediction_time

try:  # The attention readouts need PyTorch; everything else works without it.
    from .attention import (
        AttentionTrainer,
        DeepLinearAttention,
        LinearAttention,
        NodeTrajectoryAttention,
        NodeTrajectoryAttentionTrainer,
        create_node_trajectory_dataset,
        create_node_trajectory_windows,
    )
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "Reservoir",
    "DynamicalSystemSimulator",
    "LorenzSimulator",
    "RidgeTrainer",
    "AttentionTrainer",
    "NodeTrajectoryAttentionTrainer",
    "LinearAttention",
    "DeepLinearAttention",
    "NodeTrajectoryAttention",
    "create_node_trajectory_windows",
    "create_node_trajectory_dataset",
    "mse",
    "nrmse",
    "normalized_error",
    "valid_prediction_time",
]
