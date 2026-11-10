from pathlib import Path

from stable_baselines3 import PPO

from lunar_lander.evaluation.evaluator import evaluate_agent


def main():
    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root
        / "models"
        / "ppo_lr_1e4"
        / "final_model"
    )
    model = PPO.load(str(model_path))

    results = evaluate_agent(
        model=model,
        episodes=100,
        seed=100,
    )

    print("\n========== EVALUATION RESULTS ==========")

    for metric, value in results.items():
        print(f"{metric}: {value}")

    print("========================================")


if __name__ == "__main__":
    main()