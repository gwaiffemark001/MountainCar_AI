"""
Mountain Car AI - Q-learning starter project.

Run:
    python mountain_car.py

The program trains an agent using Q-learning, saves its Q-table,
and then displays the trained car.
"""

import numpy as np
import gymnasium as gym


# -------------------- SETTINGS --------------------
ENV_NAME = "MountainCar-v0"
EPISODES = 20_000
MAX_STEPS = 200

# Divide the continuous position and velocity into smaller groups.
POSITION_BINS = 40
VELOCITY_BINS = 40

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995

Q_TABLE_FILE = "q_table.npy"


def make_bins():
    """Create the boundaries used to convert observations into table indexes."""
    position_low, position_high = -1.2, 0.6
    velocity_low, velocity_high = -0.07, 0.07

    position_bins = np.linspace(
        position_low, position_high, POSITION_BINS - 1
    )
    velocity_bins = np.linspace(
        velocity_low, velocity_high, VELOCITY_BINS - 1
    )
    return position_bins, velocity_bins


def get_state(observation, bins):
    """Convert car position and velocity into discrete Q-table indexes."""
    position, velocity = observation
    position_index = np.digitize(position, bins[0])
    velocity_index = np.digitize(velocity, bins[1])
    return position_index, velocity_index


def choose_action(q_table, state, epsilon, action_space):
    """Explore with probability epsilon; otherwise choose the best known action."""
    if np.random.random() < epsilon:
        return action_space.sample()
    return int(np.argmax(q_table[state]))


def train_agent():
    """Train the car using the Q-learning update rule."""
    env = gym.make(ENV_NAME)
    bins = make_bins()

    q_table = np.zeros(
        (POSITION_BINS, VELOCITY_BINS, env.action_space.n),
        dtype=np.float32,
    )

    epsilon = EPSILON_START
    rewards = []

    print("Training started. This may take a few minutes.")

    for episode in range(1, EPISODES + 1):
        observation, info = env.reset()
        state = get_state(observation, bins)
        total_reward = 0

        for step in range(MAX_STEPS):
            action = choose_action(q_table, state, epsilon, env.action_space)

            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = get_state(next_observation, bins)

            # Q-learning update:
            # new value = old value + learning rate * prediction error
            best_next_value = np.max(q_table[next_state])
            old_value = q_table[state][action]
            new_value = old_value + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * best_next_value - old_value
            )
            q_table[state][action] = new_value

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % 1000 == 0:
            average_reward = np.mean(rewards[-1000:])
            print(
                f"Episode {episode:5d}/{EPISODES} | "
                f"Average reward (last 1000): {average_reward:.2f} | "
                f"Exploration: {epsilon:.3f}"
            )

    env.close()
    np.save(Q_TABLE_FILE, q_table)
    print(f"\nTraining complete. Q-table saved as {Q_TABLE_FILE}.")
    return q_table, bins


def watch_agent(q_table, bins, episodes=3):
    """Display the trained agent playing the Mountain Car game."""
    env = gym.make(ENV_NAME, render_mode="human")

    print("\nOpening the trained Mountain Car simulation...")
    for episode in range(1, episodes + 1):
        observation, info = env.reset()
        state = get_state(observation, bins)
        total_reward = 0

        for step in range(MAX_STEPS):
            # No random exploration during testing.
            action = int(np.argmax(q_table[state]))
            observation, reward, terminated, truncated, info = env.step(action)
            state = get_state(observation, bins)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Test episode {episode}: reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    trained_q_table, state_bins = train_agent()
    watch_agent(trained_q_table, state_bins)
