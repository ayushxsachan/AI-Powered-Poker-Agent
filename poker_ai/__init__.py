"""Poker AI research platform.

The package is intentionally modular: the engine can run without the
reinforcement-learning stack, while training scripts opt into PyTorch,
Gymnasium, Stable-Baselines3, and visualization dependencies when installed.
"""

__all__ = ["agents", "analytics", "engine", "training", "ui"]

