import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        step_penalty: float = 0.01,
    ):
        super().__init__(env)

        if step_penalty < 0:
            raise ValueError("step_penalty must be non-negative")

        self.step_penalty = step_penalty

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        shaped_reward = reward - self.step_penalty

        return (
            observation,
            shaped_reward,
            terminated,
            truncated,
            info,
        )