"""Focused checks for the reusable assignment algorithms."""

import numpy as np

from learning_algorithms import (
    ConstrainedQLearning,
    FunctionApproxQLearning,
    TabularQLearning,
    fourier_feature_extractor,
)


def main():
    features = fourier_feature_extractor([0.0, 0.0], order=2)
    assert features.shape == (9,)
    assert np.allclose(features, 1.0)

    tabular = TabularQLearning(actions=[0, 1, 2], learning_rate=1.0)
    tabular.incorporate_feedback("s", 1, 5, "next", terminal=True)
    assert tabular.q_values[("s", 1)] == 5

    function = FunctionApproxQLearning(actions=[0, 1, 2], learning_rate=1.0)
    function.incorporate_feedback([0.0, 0.0], 1, 5, [0.0, 0.0], terminal=True)
    assert function.q_value([0.0, 0.0], 1) == 80

    constrained = ConstrainedQLearning(actions=[0, 1, 2])
    actions = constrained.valid_actions("s", position=-0.5, velocity=0.0649)
    assert actions == [0, 1]
    assert constrained.get_action("s", position=-0.5, velocity=0.0649) in actions

    print("learning algorithm checks: passed")


if __name__ == "__main__":
    main()
