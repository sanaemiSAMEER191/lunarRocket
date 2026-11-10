# PPO Learning Rate Experiment Analysis

## 1. Experimental Objective

To investigate how changing the PPO learning rate affects the learning process and final landing performance in Gymnasium LunarLander-v3.

## 2. Experimental Configurations

| Parameter | Baseline | Experiment 2 |
|---|---:|---:|
| Algorithm | PPO | PPO |
| Learning rate | 3e-4 | 1e-4 |
| Training timesteps | 3,000,000 | 3,000,000 |
| Random seed | 42 | 42 |
| Evaluation episodes | 100 | 100 |
| Evaluation seeds | 100–199 | 100–199 |

All other PPO hyperparameters and network architecture were kept unchanged.

## 3. Evaluation Results

| Metric | Baseline | Experiment 2 |
|---|---:|---:|
| Mean reward | 172.086 | 110.515 |
| Reward standard deviation | 109.523 | 121.153 |
| Mean episode length | 308.910 | 293.180 |
| Success rate | 75% | 56% |
| Successful landings | 75 | 56 |

## 4. Learning Curve Analysis

The TensorBoard `rollout/ep_rew_mean` graph shows different learning patterns:

- The baseline configuration demonstrates stronger reward improvement during the later stages of training.
- The lower learning rate exhibits considerable fluctuations and weaker improvement across much of the training process.
- Towards the end, the lower learning rate also shows improvement, but its final performance remains below the baseline.

## 5. Conclusion

The baseline PPO configuration with a learning rate of 3e-4 outperformed the configuration using 1e-4.

It achieved:
- 19 percentage points higher landing success rate.
- 61.571 points higher mean evaluation reward.
- Lower reward variability.

Therefore, 3e-4 is the better-performing learning rate among the two configurations tested under the given experimental conditions.

This does not establish it as the globally optimal learning rate, but it provides empirical evidence supporting its selection for this project.