import numpy as np

from poker_ai.engine.poker_env import OBSERVATION_SIZE, PokerEnv


def test_env_reset_and_step():
    env = PokerEnv(seed=7)
    observation, info = env.reset(seed=7)

    assert observation.shape == (OBSERVATION_SIZE,)
    assert "legal_actions" in info

    mask = env.legal_action_mask()
    action = int(np.flatnonzero(mask)[0])
    next_observation, reward, terminated, truncated, next_info = env.step(action)

    assert next_observation.shape == (OBSERVATION_SIZE,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "action_mask" in next_info

