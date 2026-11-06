# MountainCar_AI

A beginner-friendly Python project that teaches a simulated car to climb a hill using **Reinforcement Learning (Q-learning)**.

## Person 1: Environment design

Person 1 is responsible for the environment model used by the learning agent.
The environment is `MountainCar-v0` from Gymnasium:

- **State:** `(position, velocity)`. Position is in `[-1.2, 0.6]` and velocity
  is in `[-0.07, 0.07]`.
- **Actions:** `0` pushes left, `1` does nothing, and `2` pushes right.
- **Goal:** reach the flag at position `0.5` or higher.
- **Reward:** `-1` for every environment step, which encourages the agent to
  reach the goal in as few steps as possible.
- **Episode limit:** an episode ends when the goal is reached or after 200
  steps.
- **Discretization:** the continuous position and velocity are divided into 40
  bins each. The resulting Q-table has shape `(40, 40, 3)`, representing
  4,800 state-action values.

The reusable environment definitions are in `environment.py`. The
`make_bins`, `get_state`, and `create_q_table` functions define how continuous
observations become Q-table indexes. The learning algorithm in
`mountain_car.py` uses those definitions without changing the environment
rules.

## 1. What you need

- Python 3.10 or newer
- Visual Studio Code
- An internet connection for installing the libraries

## 2. Open the project

1. Extract `MountainCar_AI.zip`.
2. Open VS Code.
3. Select **File > Open Folder**.
4. Choose the extracted `MountainCar_AI` folder.

## 3. Open the VS Code terminal

Select **Terminal > New Terminal**.

Check Python:

```bash
python --version
```

If that command does not work on Windows, try:

```bash
py --version
```

## 4. Create a virtual environment (recommended)

On Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

If you use PowerShell and activation is blocked, you can use Command Prompt in VS Code, or ask for help before changing execution-policy settings.

## 5. Install the required libraries

```bash
python -m pip install -r requirements.txt
```

If `python` is not recognized, use `py` instead:

```bash
py -m pip install -r requirements.txt
```

## 6. Run the project

```bash
python mountain_car.py
```

The program trains the Q-learning agent. Training can take a few minutes. It prints progress every 1,000 episodes, saves the learned table as `q_table.npy`, and then opens a window showing the trained car.

## 7. Understand the project

- **Agent:** the computer program learning to control the car.
- **Environment:** the Mountain Car game.
- **State:** the car's position and velocity.
- **Actions:** push left, do nothing, or push right.
- **Reward:** the feedback received after an action. MountainCar gives a penalty on each step, encouraging the agent to reach the flag sooner.
- **Q-table:** stores the agent's learned estimates of how useful each action is in each discretized state.
- **Exploration:** trying actions to learn.
- **Exploitation:** choosing the action currently believed to be best.

## 8. Important files

- `mountain_car.py` — the complete training and testing program.
- `requirements.txt` — Python libraries needed by the project.
- `q_table.npy` — created automatically after training; contains the learned Q-values.

## Troubleshooting

- **`python` is not recognized:** try `py` in place of `python`.
- **`No module named gymnasium`:** activate `.venv`, then run `python -m pip install -r requirements.txt`.
- **The simulation window does not open:** make sure you are running the program on your computer with a desktop display, not in a headless terminal.
- **Training is taking time:** this is normal; the starter setting is 20,000 episodes.

## Learning note

This project is a starting implementation. Q-learning results can vary with settings and random exploration. If the car does not reliably reach the flag, we can tune the number of episodes, state bins, and learning parameters together.
