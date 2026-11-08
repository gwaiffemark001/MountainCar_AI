import numpy as np

ENV_NAME = "MountainCar-v0"
MAX_STEPS = 200
POSITION_BINS = 40
VELOCITY_BINS = 40
ACTION_COUNT = 3


def make_bins():
    """Create position and velocity boundaries."""
    position_bins = np.linspace(-1.2, 0.6, POSITION_BINS - 1)
    velocity_bins = np.linspace(-0.07, 0.07, VELOCITY_BINS - 1)
    return position_bins, velocity_bins


def get_state(observation, bins):
    """Convert position and velocity into table indexes."""
    position, velocity = observation
    return (
        np.digitize(position, bins[0]),
        np.digitize(velocity, bins[1]),
    )


def create_q_table():
    """Create an empty Q-table."""
    return np.zeros((POSITION_BINS, VELOCITY_BINS, ACTION_COUNT))
