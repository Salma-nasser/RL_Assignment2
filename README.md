# RL_Assignment2

The goal of this assignment is to implement DQN and DDQN models using PyTorch, train RL agents on multiple classical environments, and use Weights & Biases to track the performed experiments.

This repository now includes a minimal training and evaluation pipeline integrated with WandB, plus video recording support.

Files added in this update:

- `rl/agents.py` — DQN and DDQN agent implementations (networks, replay memory, optimize).
- `train.py` — CLI script to train an agent and log metrics to Weights & Biases (WandB). Supports video recording via `gym.wrappers.RecordVideo`.
- `run_tests.py` — Run a saved model for multiple episodes (defaults to 100) and log episode durations.
- `requirements.txt` — minimal dependencies.

## Quick start

1. Create a virtual environment and install requirements:

```powershell
python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2. Train a DQN on CartPole and record video:

```powershell
python train.py --env CartPole-v1 --algo dqn --episodes 200 --record-video --project my-wandb-project --entity YOUR_WANDB_ENTITY
```

3. Run 100 test episodes using the saved model and log durations to WandB and CSV:

```powershell
python run_tests.py --env CartPole-v1 --model models/CartPole-v1_dqn.pt --episodes 100 --project my-wandb-project --entity YOUR_WANDB_ENTITY
```

## Hyperparameter search

You can run multiple training runs with different hyperparameters by calling `train.py` with modified flags. For automated sweeps use WandB Sweeps — create a sweep configuration in the WandB UI and launch agents with the `--project` and `--entity` flags.

## Recording videos

`--record-video` uses Gymnasium's `RecordVideo` wrapper and stores videos in the `videos/` folder.

## Notes and next steps

- The training loops are intentionally minimal and modular. You can import functions from `train.py` or `rl.agents` into notebooks.
- Consider adding better checkpointing, eval during training, and a simple hyperparameter grid runner if you want to automate experiments locally.
