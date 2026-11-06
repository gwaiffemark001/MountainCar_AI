# MountainCar_AI

A beginner-friendly Python project that teaches a simulated car to climb a hill using **Reinforcement Learning (Q-learning)**.

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
