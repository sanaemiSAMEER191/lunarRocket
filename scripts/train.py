from pathlib import Path

from lunar_lander.training.trainer import train_baseline


def main():
    project_root = Path(__file__).resolve().parents[1]

    latest_checkpoint = (
        project_root
        / "models"
        / "checkpoints"
        / "ppo_lunar_lander_1705240_steps.zip"
    )

    train_baseline(
        total_timesteps=1_000_000,
        seed=42,
        resume_from=str(latest_checkpoint),
    )


if __name__ == "__main__":
    main()