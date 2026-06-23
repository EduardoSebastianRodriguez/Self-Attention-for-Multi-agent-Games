"""Reproduce Fig. 3 of the paper from saved loss curves.

By default this script reads the loss curves from `precomputed_losses/`
(checked into the repo). If you have re-trained with
`train_ours_navigation.py`, the freshly produced `.npy` files in the
current working directory take precedence.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
PRECOMPUTED = os.path.join(HERE, "precomputed_losses")


def _load(name):
    cwd_path = name + ".npy"
    if os.path.isfile(cwd_path):
        return np.load(cwd_path)
    return np.load(os.path.join(PRECOMPUTED, name + ".npy"))


# Load data
log_loss_ours = _load("log_loss_pg_state")
log_loss_ilqr = _load("Js")

# Pad iLQR to match length
for iteration in range(len(log_loss_ilqr), len(log_loss_ours)):
    log_loss_ilqr = np.concatenate((log_loss_ilqr, [log_loss_ilqr[-1]]))  

# LaTeX + font configuration
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 20,
    "font.size": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

# Create figure and axes
fig, ax = plt.subplots()

# Plot data
ax.plot(log_loss_ours, linewidth=5, label="ours")
ax.plot(log_loss_ilqr, '--', linewidth=5, label="DP-iLQR")

# Axes labels with bold LaTeX style
ax.set_xlabel('iterations')
ax.set_ylabel('cost')

# Log scale for x-axis
ax.set_xscale("log")

# Scientific notation on y-axis
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
ax.yaxis.set_major_formatter(formatter)

# Grid
ax.grid()

# Legend styling
legend = ax.legend(frameon=True)
legend.get_frame().set_alpha(0.8)
legend.get_frame().set_facecolor('#F0F0F0')  # Light gray

# Ticks bold
ax.tick_params(axis='both', which='major', labelsize=20, width=1.5)
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontweight('bold')

# Layout
plt.tight_layout()
plt.show()
