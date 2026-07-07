# Sec. IV-C — Pursuit & Evasion in BenchMARL

This experiment reproduces **Figs. 6 and 7** of the paper: nonparametric
binomial paired tests over 500 episodes of the Simple-Tag pursuit-evasion
game, comparing our attention-based policy against MLP and GAT (graph
attention network) policies trained with **MAPPO** and **MADDPG** in
[BenchMARL](https://github.com/facebookresearch/BenchMARL).

The scenario is 2 teams × 3 holonomic agents, `bound = 1.0`, two obstacles
of radius 0.2 and a 1 m communication radius (see
[`benchmarl/conf/task/vmas/simple_tag_pos.yaml`](benchmarl/conf/task/vmas/simple_tag_pos.yaml)).
Policies are trained for `3 × 10⁶` environment steps with the default
MAPPO / MADDPG hyperparameters.

## What is in here

This folder bundles the modified BenchMARL fork used for the paper (see
**Attribution** below). Only the files that constitute our contribution are
listed here; everything else is upstream BenchMARL.

- `benchmarl/models/attention.py` — our policy parameterization
  `SelfAttentionNonlinearPolicy` (the `AttentionConfig` model), the masked
  `AttentionLayer` / `AttentionGain` blocks and the graph builder.
- `benchmarl/attention_exp.py` — driver for **our** attention policy (MAPPO).
- `benchmarl/gnn_exp.py` — GAT (GATv2) baseline (MAPPO).
- `benchmarl/gnn_exp_maddpg.py` — GAT (GATv2) baseline (MADDPG).
- `benchmarl/environments/vmas/simple_tag_pos.py` — the modified Simple-Tag
  scenario (adds `pos` / `vel` observation keys and a per-agent `catch`
  info flag), registered as `VmasTask.SIMPLE_TAG_POS`.
- `benchmarl/experiment/experiment.py` — the `_evaluation_loop` is extended
  with `get_dist_metrics` (min inter-team distance) and `get_time_metrics`
  (number/timing of catches), which produce the quantities compared in
  Figs. 6–7.
- `benchmarl/eval_results.py` — upstream `marl-eval` plotting helpers used
  for the aggregate figures.

The MLP baseline uses BenchMARL's stock `MlpConfig` policy on the same task
(swap `AttentionConfig` for `MlpConfig` in a driver, or run via
`benchmarl/run.py model=mlp`).

## Installation

Python 3.11 is recommended. From this folder:

```bash
pip install -e .
pip install -r requirements.txt
```

`torch_cluster` must match your installed `torch`/CUDA build; on CPU the
graph is built on the host, so a CPU wheel is fine.

## Reproduce the figures

Training produces a checkpoint every run; evaluation replays 500 episodes
and prints/plots the distance and catch metrics.

```bash
cd benchmarl

# Train (3e6 steps) — repeat per method; writes checkpoints under outputs/
python attention_exp.py    --train
python gnn_exp.py          --train
python gnn_exp_maddpg.py   --train

# Evaluate a trained checkpoint (dist + time metrics, with plots)
python attention_exp.py  --checkpoint outputs/self_attention_simple_tag_mappo/checkpoints/checkpoint_3000000.pt
python gnn_exp.py        --checkpoint outputs/gnn_simple_tag_mappo/checkpoints/checkpoint_3000000.pt
python gnn_exp_maddpg.py --checkpoint outputs/gnn_simple_tag_maddpg/checkpoints/checkpoint_3000000.pt
```

Each driver takes `--output <folder>` to change where checkpoints and
evaluation artifacts are written (default: `outputs/<method>/`). The
500-episode count and other evaluation settings live in
`benchmarl/conf/experiment/base_experiment.yaml`.

> **Note.** No pre-trained checkpoints are shipped here — reproduction
> requires training first (several hours per method). The trained weights
> used for the Robotarium deployment in Sec. IV-D are available under
> [`../iv_d_robotarium/`](../iv_d_robotarium).

## Attribution

This experiment is built on
[**BenchMARL**](https://github.com/facebookresearch/BenchMARL) © Meta
Platforms, Inc. (MIT). The bundled fork is a modified copy of BenchMARL;
its license is kept here as [`LICENSE_benchmarl.txt`](LICENSE_benchmarl.txt).
The Simple-Tag scenario derives from
[VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) ©
ProrokLab (MIT).
