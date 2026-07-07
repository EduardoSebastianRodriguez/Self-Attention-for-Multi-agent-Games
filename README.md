# Policy Gradient with Self-Attention for Model-Free Distributed Nonlinear Multi-Agent Games

Code for the IROS 2026 paper

> **Policy Gradient with Self-Attention for Model-Free Distributed
> Nonlinear Multi-Agent Games**
> Eduardo Sebastián, Maitrayee Keskar, Eeman Iqbal, Eduardo Montijano,
> Carlos Sagüés, Nikolay Atanasov.

This repository is organized by paper section. Each experiment lives in
its own folder, has its own README and its own reproduction
instructions.

| Folder | Paper section | Reproduces |
| ------ | ------------- | ---------- |
| [`experiments/iv_a_lqr/`](experiments/iv_a_lqr) | IV-A: Distributed linear quadratic regulation | **Fig. 2** |
| [`experiments/iv_b_navigation/`](experiments/iv_b_navigation) | IV-B: Multi-agent nonlinear navigation | **Figs. 3, 4, 5** and **Table I** |
| [`experiments/iv_c_benchmarl/`](experiments/iv_c_benchmarl) | IV-C: Pursuit-evasion in BenchMARL | **Figs. 6, 7** |
| [`experiments/iv_d_robotarium/`](experiments/iv_d_robotarium) | IV-D: Zero-shot Robotarium deployment | **Figs. 8, 9** |

## Quick reproduction (figures only)

Each experiment ships precomputed loss curves / trained weights so the
paper figures can be regenerated without retraining:

```bash
# Fig. 2 — LQR
cd experiments/iv_a_lqr && python case_1_linear_system_plot.py

# Fig. 3 — navigation training curves
cd experiments/iv_b_navigation && python plot_training_curves.py

# Figs. 8-9 — Robotarium deployment (in-simulator)
cd experiments/iv_d_robotarium && python simple_tag_robotarium.py
```

Full retraining instructions are in each subfolder's README.

## Installation

This repo is intentionally split per experiment to avoid forcing a
single environment on you. The shared dependencies are:

```bash
pip install -r requirements.txt
```

Some experiments need extras (DP-iLQR builds a Cython extension; the
Robotarium simulator needs `cvxopt`). Refer to each subfolder's README.

## Repository layout

```
.
├── README.md                ← (this file)
├── LICENSE
├── requirements.txt         ← shared deps (torch, numpy, matplotlib, …)
├── .gitignore
└── experiments/
    ├── iv_a_lqr/            ← Fig. 2
    ├── iv_b_navigation/     ← Figs. 3, 4, 5 + Table I (bundles a copy of dp-ilqr)
    ├── iv_c_benchmarl/      ← Figs. 6, 7 (bundles a modified fork of BenchMARL)
    └── iv_d_robotarium/     ← Figs. 8, 9 (bundles a copy of robotarium_python_simulator)
```

## Attribution

- The **DP-iLQR** baseline used in Sec. IV-B is bundled as
  `experiments/iv_b_navigation/dpilqr/`. Original work © Zach Williams
  (MIT, <https://github.com/labicon/dp-ilqr>). License at
  `experiments/iv_b_navigation/LICENSE.txt`.
- The **Robotarium Python Simulator** used in Sec. IV-D is bundled as
  `experiments/iv_d_robotarium/rps/`. © Georgia Institute of Technology
  (MIT, <https://github.com/robotarium/robotarium_python_simulator>).
  License at `experiments/iv_d_robotarium/LICENSE_rps.txt`.
- The Sec. IV-C experiment is built on **BenchMARL**, bundled as a modified
  fork under `experiments/iv_c_benchmarl/benchmarl/`. © Meta Platforms, Inc.
  (MIT, <https://github.com/facebookresearch/BenchMARL>). License at
  `experiments/iv_c_benchmarl/LICENSE_benchmarl.txt`.
- The IV-C and IV-D pursuit-evasion scenarios are a light modification of the
  Simple Tag environment from
  [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) ©
  ProrokLab (MIT).

## Citation

```bibtex
@inproceedings{Sebastian2026PolicyGradientSelfAttention,
  title  = {Policy Gradient with Self-Attention for Model-Free Distributed Nonlinear Multi-Agent Games},
  author = {Sebasti\'an, Eduardo and Keskar, Maitrayee and Iqbal, Eeman and Montijano, Eduardo and Sag\"u\'es, Carlos and Atanasov, Nikolay},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year   = {2026}
}
```
