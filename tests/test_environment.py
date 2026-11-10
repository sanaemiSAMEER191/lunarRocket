import numpy as np

from lunar_lander.environment.factory import create_environment


def test_environment_initialization():
    env = create_environment()

    try:
        observation, info = env.reset(seed=42)

        assert observation.shape == (8,)
        assert env.action_space.n == 4
        assert isinstance(info, dict)
        assert np.isfinite(observation).all()

    finally:
        env.close()


def test_environment_step():
    env = create_environment()

    try:
        observation, info = env.reset(seed=42)

        action = env.action_space.sample()

        next_observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        assert next_observation.shape == (8,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    finally:
        env.close()