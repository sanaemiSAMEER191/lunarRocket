import gymnasium as gym


def create_environment(
    render_mode: str | None = None,
) -> gym.Env:
    """
    Create and return a LunarLander-v3 environment.

    Args:
        render_mode: Rendering mode ('human', 'rgb_array', or None).

    Returns:
        Configured Gymnasium environment.
    """
    return gym.make(
        "LunarLander-v3",
        render_mode=render_mode,
    )