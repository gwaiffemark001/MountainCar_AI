"""Value iteration for a small finite Markov decision process.

This is the first guided assignment component. It is intentionally separate
from the Gymnasium Mountain Car code because value iteration needs a known,
finite transition model.
"""

from collections.abc import Callable, Iterable


def value_iteration(
    states: Iterable[int],
    actions: Iterable[str],
    transitions: Callable[[int, str], Iterable[tuple[float, int, float]]],
    discount: float = 1.0,
    iterations: int = 2,
) -> dict[int, float]:
    """Estimate optimal state values using repeated Bellman updates.

    ``transitions(state, action)`` must yield
    ``(probability, next_state, reward)`` tuples.
    """
    values = {state: 0.0 for state in states}
    states = list(values)
    actions = list(actions)

    for _ in range(iterations):
        updated = {}

        for state in states:
            action_values = []
            for action in actions:
                total = 0.0
                for probability, next_state, reward in transitions(state, action):
                    total += probability * (reward + discount * values[next_state])
                action_values.append(total)
            updated[state] = max(action_values, default=0.0)

        values = updated

    return values


def best_action(
    state: int,
    actions: Iterable[str],
    transitions: Callable[[int, str], Iterable[tuple[float, int, float]]],
    values: dict[int, float],
    discount: float = 1.0,
) -> str | None:
    """Return the action with the highest one-step look-ahead value."""
    best = None
    best_value = float("-inf")

    for action in actions:
        action_value = sum(
            probability * (reward + discount * values[next_state])
            for probability, next_state, reward in transitions(state, action)
        )
        if action_value > best_value:
            best = action
            best_value = action_value

    return best
