import numpy as np


def calculate_position_shaping(
    observation: np.ndarray,
) -> float:
    """
    Calculate the positional shaping score used by LunarLander.
    """
    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = (
        observation
    )

    position_penalty = -100 * np.sqrt(x**2 + y**2)
    velocity_penalty = -100 * np.sqrt(vx**2 + vy**2)
    orientation_penalty = -100 * abs(angle)

    landing_bonus = 10 * (left_leg + right_leg)

    return float(
        position_penalty
        + velocity_penalty
        + orientation_penalty
        + landing_bonus
    )