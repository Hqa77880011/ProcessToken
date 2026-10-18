# ProcessToken

Implementation of **ProcessToken: Process-Graph-Guided Masked State-Space Pretraining for Cross-Tool Semiconductor Process Modeling**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Generate synthetic process-event sequences with variable IDs, values, observation flags, timestamps, stages and recipe IDs.

```bash
processtoken demo --output data/generated.npz --count 96 --config config.json --seed 13
processtoken prepare --input data/generated.npz --output data/prepared
```

`x[N,T,6]`: variable ID, value, observation flag, timestamp, online stage and recipe ID; `y[N]`: fault class; `tool[N]`: source tool label. Vocabulary sizes and decision stage are in `config.json`.

Numeric NPZ input uses `group` to keep related samples together. `prepare` writes train/validation/calibration/test archives from ordered groups (60/15/10/15), or honors explicit `split` codes 0/1/2/3.

## Training and inference

```bash
processtoken train --input data/prepared --output runs/pretrain --config configs/pretrain.json --epochs 20 --batch-size 16 --seed 13
processtoken train --input data/prepared --output runs/full --config config.json --init runs/pretrain/model.pt --variant full --epochs 50 --batch-size 16 --seed 13
```

Use `--device cuda` to select CUDA; CPU is the default. Keep the configuration, splits and seed fixed across comparisons.

## Evaluation

```bash
processtoken evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
```

## Ablations

Variants: `full`, `baseline`, `random_mask`, `no_stage_memory`, `no_separation`. Reuse the prepared data and training seed:

```bash
processtoken train --input data/prepared --output runs/baseline --config config.json --init runs/pretrain/model.pt --variant baseline --epochs 50 --batch-size 16 --seed 13
processtoken evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

For pretrained comparisons, initialize each fine-tuning run with `--init runs/pretrain/model.pt` and use the same 50-epoch fine-tuning budget.

## Outputs

- `data/prepared/*.npz`: grouped data splits.
- `runs/full/model.pt`: validation-selected parameters and model configuration.
- `runs/full/history.json` and `training.json`: objectives and run settings.
- `runs/full/evaluation/predictions.npz` and `metrics.json`: predictions/mechanism arrays and computed metrics.

```bash
processtoken plot --input runs --output runs/figures
```

`runs/figures/*.png` compares saved evaluation metrics. All data, checkpoint and output paths are CLI arguments.
