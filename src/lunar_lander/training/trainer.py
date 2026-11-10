from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from lunar_lander.agents.ppo_agent import create_ppo_agent
from lunar_lander.environment.factory import create_environment
from lunar_lander.rewards.shaping import RewardShapingWrapper


def train_baseline(
    total_timesteps: int = 1_000_000,
    seed: int = 42,
    resume_from: str | None = None,
    learning_rate: float = 3e-4,
    experiment_name: str = "baseline",
) -> PPO:
    """Train or resume the baseline PPO agent."""

    project_root = Path(__file__).resolve().parents[3]
    experiment_dir = project_root / "experiments" / experiment_name

    model_dir = project_root / "models" / experiment_name
    checkpoint_dir = model_dir / "checkpoints"
    log_dir = experiment_dir / "logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)


    base_env = create_environment()

    shaped_env = RewardShapingWrapper(
        base_env,
        step_penalty=0.01,
    )

    env = Monitor(
        shaped_env,
        filename=str(log_dir / "monitor.csv"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=15_000,
        save_path=str(checkpoint_dir),
        name_prefix=f"ppo_{experiment_name}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    try:
        if resume_from:
            print(f"Resuming training from: {resume_from}")
            model = PPO.load(resume_from, env=env)
        else:
            model = create_ppo_agent(
                env=env,
                seed=seed,
                tensorboard_log=str(log_dir / "tensorboard"),
                learning_rate=learning_rate,
            )

        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            tb_log_name="PPO_Baseline",
            reset_num_timesteps=(resume_from is None),
        )

        final_model_path = model_dir / "ppo_lunar_lander_baseline"
        model.save(str(model_dir / "final_model"))

        print(f"Final model saved to: {final_model_path}")

        return model

    finally:
        env.close()