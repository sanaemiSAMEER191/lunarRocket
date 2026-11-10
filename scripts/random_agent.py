from lunar_lander.environment.factory import create_environment


def main():
    env = create_environment(render_mode="human")

    try:
        _observation, _info = env.reset(seed=42)
        episode_reward = 0.0
        episode_count = 0

        while True:
            action = env.action_space.sample()

            _observation, reward, terminated, truncated, _info = (
                env.step(action)
            )

            episode_reward += reward

            if terminated or truncated:
                episode_count += 1

                print(
                    f"Episode: {episode_count} | "
                    f"Reward: {episode_reward:.2f}"
                )

                _observation, _info = env.reset()
                episode_reward = 0.0

    except KeyboardInterrupt:
        print("\nSimulation stopped.")

    finally:
        env.close()


if __name__ == "__main__":
    main()