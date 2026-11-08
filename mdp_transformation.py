"""Reduce a discounted MDP to an equivalent undiscounted MDP."""

OUT = "OUT"


def transformed_transition_probability(
    original_probability: float,
    discount: float,
    next_state,
):
    """Return the probability of a transformed transition.

    The continuation probability is scaled by ``discount``. The caller can
    add the missing ``1 - discount`` probability to ``OUT``.
    """
    if not 0 <= discount <= 1:
        raise ValueError("discount must be between zero and one")
    if original_probability < 0:
        raise ValueError("original_probability cannot be negative")
    if next_state == OUT:
        raise ValueError("OUT is added by the transformed model")
    return discount * original_probability


def transformed_reward(original_reward: float, discount: float) -> float:
    """Scale continuation rewards so the expected immediate reward is equal."""
    if not 0 < discount <= 1:
        raise ValueError("discount must be greater than zero and at most one")
    return original_reward / discount


def explain_construction() -> str:
    """Return the construction used by this module."""
    return (
        "For each original transition, use gamma times its probability and "
        "reward R/gamma. Add 1-gamma probability to an absorbing OUT state "
        "with zero reward. The undiscounted Bellman equation then matches the "
        "original discounted equation."
    )
