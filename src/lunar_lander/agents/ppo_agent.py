
from gymnasium import Env
from stable_baselines3 import PPO


def create_ppo_agent(
    env: Env,
    seed: int = 42,
    tensorboard_log: str | None = None,
    learning_rate = 3e-4,
) -> PPO:
    """
    Create a PPO agent for the LunarLander environment.

    Args:
        env: Gymnasium environment.
        seed: Random seed for reproducibility.
        tensorboard_log: Directory for TensorBoard logs.

    Returns:
        Configured Stable-Baselines3 PPO model.
    """

    policy_kwargs = {
        "net_arch": {
            "pi": [64, 64],
            "vf": [64, 64],
        }
    }

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )

    return model
