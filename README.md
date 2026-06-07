# MOSSAC eVTOL Risk-Aware Path Planning

This repository provides the core implementation of MOSSAC, a risk-aware Soft Actor-Critic method for eVTOL path planning. The environment considers static obstacles, dynamic obstacles, casualty risk, and meteorological risk on a 101x101 grid map.

## Files

- `env_MOSSAC.py`: Gym-style path-planning environment, radar observation model, obstacle dynamics, and reward components.
- `MOS_SAC.py`: MOSSAC agent, replay buffer, shared feature encoder, actor network, and twin critic networks.
- `train_MOS_SAC_tensorboard.py`: training, rendering, testing, path logging, and statistics entry points.
- `generate_grid_map.py`: utility for generating obstacle-grid maps from circular obstacle definitions.
- `obstacle_grid_101x101.txt`: default static obstacle grid.
- `casualty_matrix_101x101.txt`: default casualty-risk matrix.
- `meteor_risk_101x101.txt`: default meteorological-risk matrix.
- `models/20251021_142342/mossac_final.pth`: pretrained MOSSAC checkpoint.
- `figures/20251021_142342/training_curves.png`: example training-curve output.

## Installation

```bash
pip install -r requirements.txt
```

PyTorch installation depends on your CUDA version. If needed, install PyTorch from the official PyTorch command for your platform first, then install the remaining packages.

## Usage

Optional smoke tests:

```bash
python env_MOSSAC.py
python generate_grid_map.py
python MOS_SAC.py
```

- `env_MOSSAC.py` loads the default maps and runs a short environment visualization test.
- `generate_grid_map.py` generates and visualizes a sample obstacle grid map.
- `MOS_SAC.py` runs an in-memory MOSSAC algorithm test with synthetic data.

Run the interactive training and evaluation script:

```bash
python train_MOS_SAC_tensorboard.py
```

Available modes:

- `train`: train MOSSAC and save checkpoints.
- `render`: train MOSSAC with periodic visualization.
- `test`: visualize a saved model for several episodes.
- `test_path`: record one trajectory and per-step reward components.
- `statistics`: run repeated evaluation episodes and report aggregate metrics.
- `help`: show usage information.

## TensorBoard

The training script writes TensorBoard logs to:

```text
runs/<timestamp>/
```

To view training curves, open a second terminal in the repository root and run:

```bash
tensorboard --logdir runs
```

Then open the local URL printed by TensorBoard, usually:

```text
http://localhost:6006
```

For remote servers or SSH environments, use:

```bash
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

The default scripts expect the three 101x101 text matrices to be located in the repository root.

## Pretrained Checkpoint

The included checkpoint can be used with the test, render, statistics, or path-logging modes:

```text
models/20251021_142342/mossac_final.pth
```

## License

This project is released under the MIT License.
