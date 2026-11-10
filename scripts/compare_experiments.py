
import json
from pathlib import Path

from stable_baselines3 import PPO

from lunar_lander.evaluation.evaluator import evaluate_agent


def main():
    project_root = Path(__file__).resolve().parents[1]

    experiments = {
    "Baseline (LR=3e-4)": (
        project_root / "models" / "ppo_lunar_lander_baseline"
    ),
    "Lower LR (LR=1e-4)": (
        project_root / "models" / "ppo_lr_1e4" / "final_model"
    ),
    }

    results = {}

    for name, model_path in experiments.items():
        print(f"\nEvaluating: {name}")

        model = PPO.load(str(model_path))

        metrics = evaluate_agent(
            model=model,
            episodes=100,
            seed=100,
        )

        results[name] = metrics

    print("\n========== EXPERIMENT COMPARISON ==========")

    metrics_to_display = [
        "mean_reward",
        "std_reward",
        "mean_episode_length",
        "success_rate",
        "successful_landings",
    ]

    print(f"\n{'Metric':<25}", end="")

    for name in experiments:
        print(f"{name:>25}", end="")

    print()

    for metric in metrics_to_display:
        print(f"{metric:<25}", end="")

        for name in experiments:
            print(f"{results[name][metric]:>25.3f}", end="")

        print()

    output_path = (
        project_root / "experiments" / "comparison_results.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
