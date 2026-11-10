from lunar_lander.training.trainer import train_baseline


def main():
    train_baseline(
        total_timesteps=3_000_000,
        seed=42,
        learning_rate=1e-4,
        experiment_name="ppo_lr_1e4",
    )


if __name__ == "__main__":
    main()