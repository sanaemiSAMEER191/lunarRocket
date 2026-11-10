import numpy as np

from lunar_lander.rewards.base import calculate_position_shaping


def test_position_shaping_at_ideal_state():
    observation = np.zeros(8, dtype=np.float32)

    reward = calculate_position_shaping(observation)

    assert reward == 0.0


def test_position_shaping_with_leg_contact():
    observation = np.zeros(8, dtype=np.float32)
    observation[6] = 1.0
    observation[7] = 1.0

    reward = calculate_position_shaping(observation)

    assert reward == 20.0


def test_position_shaping_penalizes_velocity():
    observation = np.zeros(8, dtype=np.float32)
    observation[2] = 1.0

    reward = calculate_position_shaping(observation)

    assert reward == -100.0