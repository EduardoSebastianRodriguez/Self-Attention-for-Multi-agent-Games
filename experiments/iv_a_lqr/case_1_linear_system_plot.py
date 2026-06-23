"""Reproduce Fig. 2 of the paper from saved loss curves.

By default this script reads the loss curves from `precomputed_losses/`
(checked into the repo). If you have re-trained with
`case_1_linear_system.py`, the freshly produced `.npy` files in the
current working directory take precedence.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

T = 100000

HERE = os.path.dirname(os.path.abspath(__file__))
PRECOMPUTED = os.path.join(HERE, "precomputed_losses")


def _load(name):
    cwd_path = name + ".npy"
    if os.path.isfile(cwd_path):
        return np.load(cwd_path)[:T]
    return np.load(os.path.join(PRECOMPUTED, name + ".npy"))[:T]


log_loss_ours = _load("loss_case_1_ours")
log_loss_optimal_lqr = _load("loss_case_1_optimal_lqr")
log_loss_furieri = _load("loss_case_1_furieri")
log_loss_fixed = _load("loss_case_1_fixed")
log_loss_pg_state = _load("loss_case_1_pg_state")

plt.rcParams.update({'font.size': 22})
plt.figure()
plt.plot(log_loss_optimal_lqr, '--', linewidth=3, label="centralized LQR")
plt.plot(log_loss_ours, linewidth=3, label="model-based ours")
plt.plot(log_loss_furieri, linewidth=3, label="furieri")
plt.plot(log_loss_fixed, linewidth=3, label="model-based ours known graph")
plt.plot(log_loss_pg_state, linewidth=3, label="model-free ours")
plt.xlabel('iterations')
plt.ylabel('cost')
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()
