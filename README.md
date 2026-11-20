# IPR: ACRA 2026 code and results

This repository collects the code and experiment results for **Integrating Action Prefix Prediction and Alignment Across Modalities for Multimodal Reinforcement Learning**. The original experiment directories are unchanged.

## Contents

- `code/dmc/rgb_position/`: five methods for RGB and position observations: `base`, `base_c`, `base_p`, `ipr_s`, and `ipr_h`.
- `code/dmc/rgb_depth/`: `base`, `ipr_s`, and `ipr_h` for RGB and depth observations.
- `code/webots/controllers/`: corresponding two-modality and three-modality Webots controllers.
- `code/webots/env/`, `worlds/`, `protos/`, and `cfg/`: the Webots environment and shared configuration.
- `results/runs/`: five runs per method and task. Each run includes its evaluation log, training log, reward array, and saved configuration and overrides. Available training summaries are also included.
- `results/summary/paper_results.csv`: final mean return and standard deviation for every method and task reported in the paper.
- `results/summary/paper_seed_results.csv`: the per-run values used for that summary.
- `results/plots/`: the five DMC task curves for each observation setting and the Webots `env1` curves, as PNG and PDF.
- `analysis/summarize_paper_results.py`: recomputes the two paper-result CSV files from the included logs using Python's standard library.
- `analysis/*_generate_result.py`: plotting and aggregation scripts retained from the experiment directories. Their original path conventions may need adjustment to regenerate plots from this layout.

`IPR-S` uses a minimum retained image crop area of 10%; `IPR-H` uses 50%. `Base-C` changes the alignment objective only, and `Base-P` changes prediction only. The latter two are included for the RGB-position component ablation.

## Recompute the reported numbers

From the release root, run:

```bash
python analysis/summarize_paper_results.py
```

For each run, the script averages the final five evaluations. It then reports the mean and population standard deviation across five runs. For RGB-depth, it uses evaluations through 300,000 steps for Catch and Swing-up and through 500,000 steps for the other tasks. It reads `mean_episode_reward` from `eval_info.log`. The Webots summaries use the `env1` navigation task. The raw CSV values retain full precision; the paper displays one decimal place.

## Code and configuration provenance

The DMC `mjpr.py`, `multi-modal_train.py`, and `config.yaml` files were taken from representative saved run snapshots for each method. The other DMC modules were copied from their corresponding source folders. The Webots `mjpr.py` files were taken from saved run snapshots; the other controller modules, environment files, and shared configuration were copied from the source folder. Every included run also has its own `run_config.yaml` and `overrides.yaml` under `results/runs/`.

These files preserve the available experiment artifacts. They are **not** a complete frozen software environment: exact dependency versions and historical copies of every shared module were not available in the runs. A fresh end-to-end training run has not been validated from this staging directory.

The separate wall-time comparison in the paper used successful compute-job records. Those records were not present in the local result folders and are not included here; the wall-time percentages cannot be recomputed from this release alone.

DMC entry points are the `multi-modal_train.py` files in each method directory. They accept an environment such as `-env walker/walk` and a seed such as `-seed 10`. Webots controllers use the matching launcher in each controller directory and the included `worlds/env1.wbt`. Select the desired controller in Webots before running it; the world file does not choose one by default.

The training code imports PyTorch, torchvision, Hydra, OmegaConf, NumPy, dm_control, dm_env, OpenCV, Matplotlib, termcolor, and wandb as applicable. Webots additionally requires its Python controller API. Exact tested package versions are not recorded here. Check the upstream license terms for code inherited from MJPR and Webots before reusing or relicensing those components.

Checkpoints, TensorBoard event files, caches, and unrelated tasks are excluded to keep the release focused and small.
