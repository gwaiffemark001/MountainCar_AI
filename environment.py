"""Mountain Car environment definitions for Person 1's project scope.

This module describes the observation space, action meanings, reward behavior,
and discretization used by the tabular Q-learning agent.
"""

from __future__ import annotations

from typing import Final

import numpy as np

ENV_NAME: Final = "MountainCar-v0"
POSITION_RANGE: Final[tuple[float, float]] = (-1.2, 0.6)
VELOCITY_RANGE: Final[tuple[float, float]] = (-0.07, 0.07)
GOAL_POSITION: Final[float] = 0.5
MAX_STEPS: Final[int] = 200
POSITION_BINS: Final[int] = 40
VELOCITY_BINS: Final[int] = 40
ACTION_COUNT: Final[int] = 3

ACTION_NAMES: Final[dict[int, str]] = {
    0: "push left",
    1: "do nothing",
    2: "push right",
}


def make_bins() -> tuple[np.ndarray, np.ndarray]:
    """Create boundaries for discretizing position and velocity."""
    position_bins = np.linspace(
        POSITION_RANGE[0],
        POSITION_RANGE[1],
        POSITION_BINS - 1,
    )
    velocity_bins = np.linspace(
        VELOCITY_RANGE[0],
        VELOCITY_RANGE[1],
        VELOCITY_BINS - 1,
    )
    return position_bins, velocity_bins


def get_state(
    observation: tuple[float, float] | np.ndarray,
    bins: tuple[np.ndarray, np.ndarray],
) -> tuple[int, int]:
    """Map a continuous ``(position, velocity)`` observation to table indexes."""
    position, velocity = observation
    position_index = int(np.digitize(position, bins[0]))
    velocity_index = int(np.digitize(velocity, bins[1]))
    return position_index, velocity_index


def create_q_table() -> np.ndarray:
    """Create an empty Q-table with one value for every state-action pair."""
    return np.zeros(
        (POSITION_BINS, VELOCITY_BINS, ACTION_COUNT),
        dtype=np.float32,
    )


def describe_reward(terminated: bool) -> int:
    """Return the Mountain Car reward for one transition.

    The environment gives -1 for each step, including the step that reaches
    the goal. The flag is reached when position is at least 0.5.
    """
    del terminated
    return -1
