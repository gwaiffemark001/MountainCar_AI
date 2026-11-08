"""Create the notebook and PDF submission package from experiment results."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).parent
DELIVERABLES = ROOT / "deliverables"
RESULTS_FILE = DELIVERABLES / "experiment_results.json"
NOTEBOOK_FILE = DELIVERABLES / "MountainCar_AI_Report.ipynb"
PDF_FILE = DELIVERABLES / "MountainCar_AI_Theory_and_Results.pdf"


def make_notebook(results: dict) -> None:
    """Write a self-contained notebook with explanations and reproducible code."""
    result_text = json.dumps(results, indent=2)
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# MountainCar AI: Reinforcement Learning Report\n",
                "\n",
                "This notebook documents the completed MountainCar-v0 project. "
                "The implementation covers value iteration, MDP transformation, "
                "model-based Monte Carlo, tabular Q-learning, Fourier features, "
                "function approximation, and constrained exploration.\n",
                "\n",
                "**Team scope:** Person 1 implemented the AI algorithm and environment "
                "integration. AI assistance was used for implementation support and "
                "testing, and the final code was reviewed and executed locally.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Environment\n",
                "\n",
                "- State: continuous `(position, velocity)`.\n",
                "- Actions: push left (`0`), no push (`1`), push right (`2`).\n",
                "- Reward: `-1` per step until the car reaches position `0.5`.\n",
                "- Horizon: 200 steps.\n",
                "- Tabular representation: 40 position bins x 40 velocity bins x 3 actions.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Core methods\n",
                "\n",
                "**Value iteration** repeatedly applies the Bellman optimality update:\n",
                "`V(s) = max_a sum_{s'} P(s'|s,a)[R(s,a,s') + gamma V(s')]`.\n",
                "\n",
                "**MDP transformation** removes discounting by multiplying continuation "
                "probabilities by `gamma`, scaling continuation rewards by `1/gamma`, "
                "and adding an absorbing `OUT` state with probability `1-gamma`.\n",
                "\n",
                "**Q-learning** updates one state-action estimate from observed transitions. "
                "Function approximation replaces the table with linear weights over "
                "Fourier features. Constrained Q-learning filters actions whose predicted "
                "velocity would exceed the safety threshold.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import json\n",
                "import sys\n",
                "sys.path.insert(0, str(Path.cwd().parent))\n",
                "results_path = Path.cwd() / 'experiment_results.json'\n",
                "results = json.loads(results_path.read_text())\n",
                "results\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Measured experiment results\n",
                "\n",
                "The following results were generated with a fixed seed. Returns are "
                "negative because the environment charges -1 per step; less negative "
                "values indicate faster successful episodes.\n",
                "\n",
                "```json\n",
                result_text,
                "\n```\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import matplotlib.pyplot as plt\n",
                "from IPython.display import Image, display\n",
                "display(Image(filename='experiment_comparison.png'))\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## How to reproduce\n",
                "\n",
                "From the repository root:\n",
                "\n",
                "```bash\n",
                ".venv/bin/python experiments.py --episodes 1000 --seed 42\n",
                ".venv/bin/python create_deliverables.py\n",
                ".venv/bin/python test_learning_algorithms.py\n",
                "```\n",
                "\n",
                "The focused checks confirm Fourier feature dimensions, terminal Q-learning "
                "updates, function-approximation updates, and constrained action filtering.\n",
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_FILE.write_text(json.dumps(notebook, indent=2) + "\n")


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.08, 0.94, title, fontsize=20, weight="bold", va="top")
    y = 0.88
    for paragraph in paragraphs:
        lines = textwrap.wrap(paragraph, width=92)
        figure.text(0.08, y, "\n".join(lines), fontsize=11, va="top", linespacing=1.5)
        y -= 0.055 + 0.026 * len(lines)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def make_pdf(results: dict) -> None:
    """Write a short theory and results report."""
    with PdfPages(PDF_FILE) as pdf:
        add_text_page(
            pdf,
            "MountainCar AI: Theory and Results",
            [
                "Project scope: Person 1 implemented the reinforcement-learning environment, "
                "agent algorithms, experiments, and evaluation workflow for Gymnasium "
                "MountainCar-v0.",
                "The state is position and velocity. The three actions push left, do nothing, "
                "or push right. Each step receives reward -1, so a higher return means the "
                "goal was reached in fewer steps. Episodes end at the goal or after 200 steps.",
                "AI assistance was used for implementation support and testing. The project "
                "was reviewed and the reported experiments were executed locally.",
            ],
        )
        add_text_page(
            pdf,
            "Algorithms",
            [
                "Value iteration applies repeated Bellman optimality updates to a known finite "
                "transition model. The implementation is generic and is tested with a small "
                "finite MDP.",
                "For the discounted-to-undiscounted MDP construction, continuation transitions "
                "use gamma times their original probability, continuation rewards use R/gamma, "
                "and the missing 1-gamma probability goes to an absorbing OUT state with zero "
                "reward. Substitution gives the original discounted Bellman equation.",
                "Model-based Monte Carlo estimates transition counts and mean rewards before "
                "running value iteration. Tabular Q-learning learns directly from samples. "
                "Function approximation uses cosine Fourier features and linear weights. "
                "Constrained Q-learning removes actions predicted to exceed the velocity limit.",
            ],
        )
        add_text_page(
            pdf,
            "Experiment configuration",
            [
                f"Seed: {results['seed']}. Episodes: {results['episodes']} for the main "
                "comparisons. The model-based run is capped at 300 episodes because it "
                "performs a value-iteration pass over the collected model.",
                "The tabular agent uses a 40 x 40 discretized state space, epsilon-greedy "
                "exploration, learning rate 0.1, and discount factor 0.99. Evaluation uses "
                "greedy actions with exploration disabled.",
            ],
        )
        figure = plt.figure(figsize=(8.27, 11.69))
        figure.text(0.08, 0.94, "Measured results", fontsize=20, weight="bold", va="top")
        y = 0.88
        for method, values in results.items():
            if not isinstance(values, dict):
                continue
            summary = (
                f"{method}: mean return={values.get('mean_return', values.get('mean_return_last_100')):.2f}, "
                f"goal rate={values.get('success_rate', values.get('success_rate_last_100')):.1%}"
            )
            figure.text(0.1, y, summary, fontsize=12, va="top")
            y -= 0.05
        figure.text(
            0.08,
            y - 0.02,
            "Less negative returns are better because every environment step has reward -1.",
            fontsize=11,
            va="top",
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"{RESULTS_FILE} does not exist; run experiments.py first"
        )
    results = json.loads(RESULTS_FILE.read_text())
    make_notebook(results)
    make_pdf(results)
    print(f"Created {NOTEBOOK_FILE}")
    print(f"Created {PDF_FILE}")


if __name__ == "__main__":
    main()
