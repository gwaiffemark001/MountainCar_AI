"""Run comparable Mountain Car experiments and save compact results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from environment import ENV_NAME, MAX_STEPS, get_state, make_bins
from learning_algorithms import (
    ConstrainedQLearning,
    FunctionApproxQLearning,
    ModelBasedMonteCarlo,
)
from mountain_car import evaluate_agent, train_agent


ROOT = Path(__file__).parent
DELIVERABLES = ROOT / "deliverables"
RESULTS_FILE = DELIVERABLES / "experiment_results.json"
PLOT_FILE = DELIVERABLES / "experiment_comparison.png"


def run_function_approximation(episodes: int, seed: int) -> dict[str, float]:
    """Train a Fourier-feature agent directly on continuous observations."""
    env = gym.make(ENV_NAME)
    agent = FunctionApproxQLearning(
        actions=[0, 1, 2],
        learning_rate=0.01,
        exploration=0.1,
    )
    rng = np.random.default_rng(seed)
    returns = []
    successes = []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            reached_goal = False
            for _ in range(MAX_STEPS):
                normalized = np.array(
                    [(observation[0] + 1.2) / 1.8, (observation[1] + 0.07) / 0.14]
                )
                action = agent.get_action(normalized)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_normalized = np.array(
                    [
                        (next_observation[0] + 1.2) / 1.8,
                        (next_observation[1] + 0.07) / 0.14,
                    ]
                )
                agent.incorporate_feedback(
                    normalized,
                    action,
                    float(reward),
                    next_normalized,
                    terminal=terminated,
                )
                observation = next_observation
                total_reward += float(reward)
                if terminated:
                    reached_goal = True
                if terminated or truncated:
                    break
            returns.append(total_reward)
            successes.append(float(reached_goal))
    finally:
        env.close()

    return {
        "episodes": float(episodes),
        "mean_return_last_100": float(np.mean(returns[-100:])),
        "success_rate_last_100": float(np.mean(successes[-100:])),
    }


def run_constrained_learning(episodes: int, seed: int) -> dict[str, float]:
    """Train tabular Q-learning while filtering unsafe velocity actions."""
    env = gym.make(ENV_NAME)
    bins = make_bins()
    agent = ConstrainedQLearning(
        actions=[0, 1, 2],
        learning_rate=0.1,
        exploration=0.1,
    )
    returns = []
    successes = []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            state = get_state(observation, bins)
            total_reward = 0.0
            reached_goal = False
            for _ in range(MAX_STEPS):
                action = agent.get_action(
                    state,
                    position=float(observation[0]),
                    velocity=float(observation[1]),
                )
                if action is None:
                    action = 1
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_state = get_state(next_observation, bins)
                agent.incorporate_feedback(
                    state, action, float(reward), next_state, terminal=terminated
                )
                observation = next_observation
                state = next_state
                total_reward += float(reward)
                if terminated:
                    reached_goal = True
                if terminated or truncated:
                    break
            returns.append(total_reward)
            successes.append(float(reached_goal))
    finally:
        env.close()

    return {
        "episodes": float(episodes),
        "mean_return_last_100": float(np.mean(returns[-100:])),
        "success_rate_last_100": float(np.mean(successes[-100:])),
    }


def run_model_based(episodes: int, seed: int) -> dict[str, float]:
    """Collect a discretized model and report its learned greedy policy."""
    env = gym.make(ENV_NAME)
    bins = make_bins()
    agent = ModelBasedMonteCarlo(
        actions=[0, 1, 2],
        discount=0.99,
        calc_value_iteration_every=max(episodes, 1),
    )
    returns = []
    successes = []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            state = get_state(observation, bins)
            total_reward = 0.0
            reached_goal = False
            for _ in range(MAX_STEPS):
                action = agent.get_action(state)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_state = get_state(next_observation, bins)
                agent.incorporate_feedback(
                    state, action, float(reward), next_state, terminal=terminated
                )
                observation = next_observation
                state = next_state
                total_reward += float(reward)
                if terminated:
                    reached_goal = True
                if terminated or truncated:
                    break
            returns.append(total_reward)
            successes.append(float(reached_goal))
    finally:
        env.close()

    return {
        "episodes": float(episodes),
        "mean_return_last_100": float(np.mean(returns[-100:])),
        "success_rate_last_100": float(np.mean(successes[-100:])),
        "model_states": float(len(agent.values)),
    }


def run_experiments(episodes: int = 1000, seed: int = 42) -> dict:
    """Run all methods and save JSON, CSV, and a comparison plot."""
    DELIVERABLES.mkdir(exist_ok=True)
    tabular, bins, history = train_agent(
        episodes=episodes,
        seed=seed,
        q_table_file=str(DELIVERABLES / "tabular_q_table.npy"),
        evaluation_interval=max(episodes // 5, 1),
    )
    tabular_return, tabular_success = evaluate_agent(
        tabular, bins, episodes=20, seed=seed + 10_000
    )
    results = {
        "seed": seed,
        "episodes": episodes,
        "tabular_q_learning": {
            "mean_return": tabular_return,
            "success_rate": tabular_success,
        },
        "function_approximation": run_function_approximation(episodes, seed + 1),
        "constrained_q_learning": run_constrained_learning(episodes, seed + 2),
        "model_based_monte_carlo": run_model_based(min(episodes, 300), seed + 3),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")

    rows = []
    for method, values in results.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "method": method,
                    "mean_return": values.get("mean_return", values.get("mean_return_last_100")),
                    "success_rate": values.get("success_rate", values.get("success_rate_last_100")),
                }
            )
    with (DELIVERABLES / "experiment_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "mean_return", "success_rate"])
        writer.writeheader()
        writer.writerows(rows)

    methods = [row["method"] for row in rows]
    returns = [row["mean_return"] for row in rows]
    success = [row["success_rate"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(methods, returns)
    axes[0].set_title("Mean return")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(methods, success)
    axes[1].set_title("Goal-reaching rate")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(PLOT_FILE, dpi=160)
    plt.close(figure)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_experiments(args.episodes, args.seed), indent=2))


if __name__ == "__main__":
    main()
