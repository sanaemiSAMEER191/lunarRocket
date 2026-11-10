
import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "models/ppo_lunar_lander_baseline.zip"

model = PPO.load(MODEL_PATH)

env = gym.make("LunarLander-v3", render_mode="human")

try:
    for episode in range(5):
        observation, info = env.reset(seed=100 + episode)
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )
            total_reward += reward

        print(
            f"Episode {episode + 1}: "
            f"Reward = {total_reward:.2f}"
        )

finally:
    env.close()
