"""Reusable reinforcement-learning algorithms for the Mountain Car project."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any

import numpy as np

from value_iteration import value_iteration


class ModelBasedMonteCarlo:
    """Estimate an MDP from experience and periodically run value iteration."""

    def __init__(
        self,
        actions,
        discount=1.0,
        calc_value_iteration_every=100,
    ):
        self.actions = list(actions)
        self.discount = discount
        self.update_every = calc_value_iteration_every
        self.steps = 0
        self.values = defaultdict(float)
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.rewards = defaultdict(float)

    def get_action(self, state):
        scores = [self.values[(state, action)] for action in self.actions]
        return self.actions[int(np.argmax(scores))]

    def incorporate_feedback(
        self,
        state,
        action,
        reward,
        next_state,
        terminal=False,
    ):
        key = (state, action)
        self.transitions[key][next_state] += 1
        self.rewards[key] += reward
        self.steps += 1

        if self.steps % self.update_every != 0:
            return

        states = {state}
        for (_, _), next_counts in self.transitions.items():
            states.update(next_counts)

        def transition_model(current_state, current_action):
            current_key = (current_state, current_action)
            counts = self.transitions[current_key]
            total = sum(counts.values())
            if not total:
                return [(1.0, current_state, 0.0)]
            mean_reward = self.rewards[current_key] / total
            return [
                (count / total, next_state, mean_reward)
                for next_state, count in counts.items()
            ]

        self.values = defaultdict(float)
        state_values = value_iteration(
            states,
            self.actions,
            transition_model,
            discount=self.discount,
            iterations=20,
        )
        for current_state in states:
            for current_action in self.actions:
                self.values[(current_state, current_action)] = max(
                    (
                        probability
                        * (reward + self.discount * state_values[next_state])
                        for probability, next_state, reward in transition_model(
                            current_state, current_action
                        )
                    ),
                    default=0.0,
                )


class TabularQLearning:
    """Q-learning over a finite state-action space."""

    def __init__(self, actions, discount=0.99, learning_rate=0.1, exploration=0.1):
        self.actions = list(actions)
        self.discount = discount
        self.learning_rate = learning_rate
        self.exploration = exploration
        self.q_values = defaultdict(float)
        self.rng = np.random.default_rng()

    def get_action(self, state):
        if self.rng.random() < self.exploration:
            return self.actions[int(self.rng.integers(len(self.actions)))]
        scores = [self.q_values[(state, action)] for action in self.actions]
        return self.actions[int(np.argmax(scores))]

    def incorporate_feedback(self, state, action, reward, next_state, terminal=False):
        current = self.q_values[(state, action)]
        next_value = 0.0
        if not terminal:
            next_value = max(
                (self.q_values[(next_state, next_action)] for next_action in self.actions),
                default=0.0,
            )
        target = reward + self.discount * next_value
        self.q_values[(state, action)] = current + self.learning_rate * (
            target - current
        )


def fourier_feature_extractor(state, order=3, scale=None):
    """Return all Fourier features up to ``order`` for a continuous state."""
    state = np.asarray(state, dtype=float)
    if scale is not None:
        state = state * np.asarray(scale, dtype=float)
    coefficients = product(range(order + 1), repeat=len(state))
    values = [np.dot(coefficient, state) for coefficient in coefficients]
    return np.cos(np.pi * np.asarray(values))


class FunctionApproxQLearning:
    """Q-learning using linear weights over extracted state features."""

    def __init__(
        self,
        actions,
        feature_extractor=fourier_feature_extractor,
        discount=0.99,
        learning_rate=0.01,
        exploration=0.1,
    ):
        self.actions = list(actions)
        self.feature_extractor = feature_extractor
        self.discount = discount
        self.learning_rate = learning_rate
        self.exploration = exploration
        self.weights = {}
        self.rng = np.random.default_rng()

    def features(self, state):
        return np.asarray(self.feature_extractor(state), dtype=float)

    def q_value(self, state, action):
        features = self.features(state)
        if action not in self.weights:
            self.weights[action] = np.zeros(len(features))
        return float(np.dot(self.weights[action], features))

    def get_action(self, state):
        if self.rng.random() < self.exploration:
            return self.actions[int(self.rng.integers(len(self.actions)))]
        values = [self.q_value(state, action) for action in self.actions]
        return self.actions[int(np.argmax(values))]

    def incorporate_feedback(self, state, action, reward, next_state, terminal=False):
        prediction = self.q_value(state, action)
        next_value = 0.0
        if not terminal:
            next_value = max(
                (self.q_value(next_state, next_action) for next_action in self.actions),
                default=0.0,
            )
        error = reward + self.discount * next_value - prediction
        self.weights[action] += self.learning_rate * error * self.features(state)


class ConstrainedQLearning(TabularQLearning):
    """Tabular Q-learning that filters actions exceeding a velocity limit."""

    def __init__(self, *args, velocity_threshold=0.065, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity_threshold = velocity_threshold

    def valid_actions(self, state, position, velocity):
        force = 0.001
        gravity = 0.0025
        valid = []
        for action in self.actions:
            next_velocity = velocity + (action - 1) * force - np.cos(
                3 * position
            ) * gravity
            if next_velocity < self.velocity_threshold:
                valid.append(action)
        return valid

    def get_action(self, state, position=None, velocity=None):
        if position is None or velocity is None:
            return super().get_action(state)
        actions = self.valid_actions(state, position, velocity)
        if not actions:
            return None
        if self.rng.random() < self.exploration:
            return actions[int(self.rng.integers(len(actions)))]
        return max(actions, key=lambda action: self.q_values[(state, action)])
