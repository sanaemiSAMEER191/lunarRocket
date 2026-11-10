from lunar_lander.training.trainer import train_baseline


def main():
    train_baseline(
        total_timesteps=3_000_000,
        seed=42,
        learning_rate=3e-4,
        experiment_name="ppo_step_penalty_001",
    )


if __name__ == "__main__":
    main()