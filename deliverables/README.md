# MountainCar AI Deliverables

This folder contains the completed submission artifacts.

- `MountainCar_AI_Report.ipynb` - notebook explanation and reproducibility guide.
- `MountainCar_AI_Theory_and_Results.pdf` - theory, MDP transformation, algorithms, and results.
- `experiment_results.json` - complete machine-readable experiment output.
- `experiment_results.csv` - compact comparison table.
- `experiment_comparison.png` - return and goal-rate chart.

The experiment results use seed `42`. Recreate them from the repository root with:

```bash
.venv/bin/python experiments.py --episodes 1000 --seed 42
.venv/bin/python create_deliverables.py
```
