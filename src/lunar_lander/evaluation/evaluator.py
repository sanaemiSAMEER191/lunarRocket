import numpy as np
from stable_baselines3 import PPO

from lunar_lander.environment.factory import create_environment


def evaluate_agent(
    model: PPO,
    episodes: int = 20,
    seed: int = 100,
) -> dict:
    """
    Evaluate a trained PPO agent over multiple episodes.
    """

    env = create_environment()

    episode_rewards = []
    episode_lengths = []
    successful_landings = 0

    try:
        for episode in range(episodes):
            observation, _info = env.reset(seed=seed + episode)

            total_reward = 0.0
            episode_length = 0
            landed_successfully = False

            while True:
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    _info,
                ) = env.step(action)

                total_reward += reward
                episode_length += 1

                if terminated or truncated:
                    landed_successfully = (
                        terminated and reward == 100
                    )
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)

            if landed_successfully:
                successful_landings += 1

    finally:
        env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "success_rate": successful_landings / episodes,
        "successful_landings": successful_landings,
        "total_episodes": episodes,
    }