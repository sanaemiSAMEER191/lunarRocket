from lunar_lander.environment.factory import create_environment


def main():
    env = create_environment()

    try:
        observation, info = env.reset(seed=42)

        print("\n--- ENVIRONMENT INFORMATION ---")
        print(f"Environment: {env.spec.id}")

        print("\n--- OBSERVATION SPACE ---")
        print(f"Shape: {env.observation_space.shape}")
        print(f"Low: {env.observation_space.low}")
        print(f"High: {env.observation_space.high}")
        print(f"Sample: {observation}")

        print("\n--- ACTION SPACE ---")
        print(f"Number of actions: {env.action_space.n}")
        print("\nAction mapping:")
        action_mapping = {
            0: "Do nothing",
            1: "Fire left orientation engine",
            2: "Fire main engine",
            3: "Fire right orientation engine",
        }

        for action, meaning in action_mapping.items():
            print(f"{action}: {meaning}")

        print("\n--- ONE ENVIRONMENT STEP ---")
        action = 2

        next_obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        print(f"Action selected: {action}")
        print(f"Next observation: {next_obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")

    finally:
        env.close()


if __name__ == "__main__":
    main()