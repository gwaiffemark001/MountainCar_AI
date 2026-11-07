"""Train, evaluate, and visualize a tabular Q-learning Mountain Car agent."""

from __future__ import annotations

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from environment import (
    ACTION_COUNT,
    ENV_NAME,
    MAX_STEPS,
    create_q_table,
    get_state,
    make_bins,
)


EPISODES = 20_000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
EVALUATION_INTERVAL = 1_000
EVALUATION_EPISODES = 10
MOVING_AVERAGE_WINDOW = 100
SEED = 42
Q_TABLE_FILE = "q_table.npy"
PLOT_FILE = "training_results.png"


def choose_action(
    q_table: np.ndarray,
    state: tuple[int, int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Explore with probability epsilon; otherwise choose a best-known action."""
    if rng.random() < epsilon:
        return int(rng.integers(ACTION_COUNT))

    state_values = q_table[state]
    best_actions = np.flatnonzero(state_values == np.max(state_values))
    return int(rng.choice(best_actions))


def evaluate_agent(
    q_table: np.ndarray,
    bins: tuple[np.ndarray, np.ndarray],
    episodes: int = EVALUATION_EPISODES,
    seed: int = SEED + 1,
) -> tuple[float, float]:
    """Return mean reward and goal-reaching rate without exploration."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")

    env = gym.make(ENV_NAME)
    episode_returns: list[float] = []
    successes: list[float] = []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            state = get_state(observation, bins)
            total_reward = 0.0
            reached_goal = False

            for _ in range(MAX_STEPS):
                action = int(np.argmax(q_table[state]))
                observation, reward, terminated, truncated, _ = env.step(action)
                state = get_state(observation, bins)
                total_reward += float(reward)

                if terminated:
                    reached_goal = True
                if terminated or truncated:
                    break

            episode_returns.append(total_reward)
            successes.append(float(reached_goal))
    finally:
        env.close()

    return float(np.mean(episode_returns)), float(np.mean(successes))


def train_agent(
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], dict[str, list[float]]]:
    """Train with Q-learning and record per-episode and evaluation metrics."""
    env = gym.make(ENV_NAME)
    bins = make_bins()
    q_table = create_q_table()
    rng = np.random.default_rng(SEED)

    if env.action_space.n != ACTION_COUNT:
        env.close()
        raise RuntimeError(
            f"{ENV_NAME} exposes {env.action_space.n} actions; "
            f"expected {ACTION_COUNT}."
        )

    epsilon = EPSILON_START
    history: dict[str, list[float]] = {
        "episode": [],
        "return": [],
        "success": [],
        "eval_episode": [],
        "eval_return": [],
        "eval_success": [],
    }

    try:
        print("Training started.")

        for episode in range(1, EPISODES + 1):
            if episode == 1:
                observation, _ = env.reset(seed=SEED)
            else:
                observation, _ = env.reset()

            state = get_state(observation, bins)
            total_reward = 0.0
            reached_goal = False

            for _ in range(MAX_STEPS):
                action = choose_action(q_table, state, epsilon, rng)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_state = get_state(next_observation, bins)

                # Do not bootstrap after reaching the terminal goal. A time-limit
                # truncation can still bootstrap from its final state.
                next_value = (
                    0.0 if terminated else float(np.max(q_table[next_state]))
                )
                old_value = float(q_table[state][action])
                td_target = float(reward) + DISCOUNT_FACTOR * next_value
                q_table[state][action] = old_value + LEARNING_RATE * (
                    td_target - old_value
                )

                state = next_state
                total_reward += float(reward)

                if terminated:
                    reached_goal = True
                if terminated or truncated:
                    break

            history["episode"].append(float(episode))
            history["return"].append(total_reward)
            history["success"].append(float(reached_goal))
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            if episode % EVALUATION_INTERVAL == 0:
                mean_eval_return, eval_success_rate = evaluate_agent(
                    q_table,
                    bins,
                    seed=SEED + episode,
                )
                history["eval_episode"].append(float(episode))
                history["eval_return"].append(mean_eval_return)
                history["eval_success"].append(eval_success_rate)

                recent_returns = history["return"][-EVALUATION_INTERVAL:]
                recent_success = history["success"][-EVALUATION_INTERVAL:]
                print(
                    f"Episode {episode:5d}/{EPISODES} | "
                    f"train return: {np.mean(recent_returns):.1f} | "
                    f"train goal rate: {np.mean(recent_success):.1%} | "
                    f"eval return: {mean_eval_return:.1f} | "
                    f"eval goal rate: {eval_success_rate:.1%} | "
                    f"epsilon: {epsilon:.3f}"
                )
    finally:
        env.close()

    np.save(Q_TABLE_FILE, q_table)
    print(f"Training complete. Q-table saved as {Q_TABLE_FILE}.")
    return q_table, bins, history


def plot_training_results(history: dict[str, list[float]]) -> None:
    """Plot smoothed training progress and greedy evaluation results."""
    returns = np.asarray(history["return"], dtype=float)
    successes = np.asarray(history["success"], dtype=float)

    if len(returns) == 0:
        raise ValueError("history contains no training episodes")

    window = min(MOVING_AVERAGE_WINDOW, len(returns))
    averages = np.ones(window, dtype=float) / window
    smoothed_returns = np.convolve(returns, averages, mode="valid")
    smoothed_success = np.convolve(successes, averages, mode="valid")
    training_episodes = np.arange(window, len(returns) + 1)
    eval_episodes = np.asarray(history["eval_episode"], dtype=float)

    figure, axes = plt.subplots(4, 1, figsize=(10, 14))

    axes[0].plot(training_episodes, smoothed_returns)
    axes[0].set(
        title="Training return (moving average)",
        xlabel="Episode",
        ylabel="Mean reward",
    )

    axes[1].plot(training_episodes, smoothed_success)
    axes[1].set(
        title=f"Training goal rate ({window}-episode moving average)",
        xlabel="Episode",
        ylabel="Goal-reaching rate",
        ylim=(0.0, 1.0),
    )

    axes[2].plot(eval_episodes, history["eval_return"], marker="o")
    axes[2].set(
        title="Greedy evaluation return",
        xlabel="Training episode",
        ylabel="Mean reward",
    )

    axes[3].plot(eval_episodes, history["eval_success"], marker="s")
    axes[3].set(
        title="Greedy evaluation goal rate",
        xlabel="Training episode",
        ylabel="Goal-reaching rate",
        ylim=(0.0, 1.0),
    )

    figure.tight_layout()
    figure.savefig(PLOT_FILE)
    print(f"Training graphs saved as {PLOT_FILE}.")
    plt.show()
    plt.close(figure)


def watch_agent(
    q_table: np.ndarray,
    bins: tuple[np.ndarray, np.ndarray],
    episodes: int = 3,
) -> None:
    """Open a rendered window to watch the greedy policy."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")

    env = gym.make(ENV_NAME, render_mode="human")

    try:
        for episode in range(1, episodes + 1):
            observation, _ = env.reset(seed=SEED + 10_000 + episode)
            state = get_state(observation, bins)
            total_reward = 0.0
            reached_goal = False

            for _ in range(MAX_STEPS):
                action = int(np.argmax(q_table[state]))
                observation, reward, terminated, truncated, _ = env.step(action)
                state = get_state(observation, bins)
                total_reward += float(reward)

                if terminated:
                    reached_goal = True
                if terminated or truncated:
                    break

            print(
                f"Test episode {episode}: reward={total_reward:.0f}, "
                f"reached goal={reached_goal}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    trained_q_table, state_bins, training_history = train_agent()
    plot_training_results(training_history)
    
