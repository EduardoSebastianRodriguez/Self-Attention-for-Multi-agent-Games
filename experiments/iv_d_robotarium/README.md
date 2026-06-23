# Sec. IV-D — Zero-shot Robotarium Deployment

This experiment reproduces the **Robotarium simulation** behavior shown
in **Figs. 8 and 9** of the paper: the MAPPO-trained pursuer / evader
policies from Sec. IV-C are loaded as-is and rolled out in the
[Georgia Tech Robotarium](https://www.robotarium.gatech.edu/) Python
simulator (non-holonomic robots with safety barrier certificates).

## Files

| File                       | Purpose                                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `simple_tag_robotarium.py` | Main deployment script. Spawns `3 pursuers + 3 evaders`, loads `actor_weights.npy`, runs 900 iterations, logs catches & distances. |
| `simple_tag.py`            | The VMAS pursuit-evasion scenario (modified Simple Tag) the policies were *trained* on. Provided here for reference / retraining. |
| `functions.py`             | `SelfAttentionNonlinearPolicy` and `AttentionGain` — same policy architecture as in IV-B, with the deployment-side forward pass.   |
| `reinforcement_functions.py` | `SquashedNormal`, `TanhTransform`, `ReplayBuffer` (helpers shared with the training side).                                    |
| `actor_weights.npy`        | Both teams' MAPPO actor weights extracted from a 3 M-step BenchMARL checkpoint — ~290 KB.                                       |
| `process_checkpoint.py`    | Helper that converts a raw BenchMARL `.pt` checkpoint (not shipped — too large) into `actor_weights.npy`.                       |
| `rps/`                     | Bundled copy of the Robotarium Python Simulator (<https://github.com/robotarium/robotarium_python_simulator>, MIT). The original `LICENSE` is preserved at `LICENSE_rps.txt`. |

## Reproduce

```bash
# In the Robotarium simulator (no real robots, no Robotarium account needed)
python simple_tag_robotarium.py
```

To deploy on the real Robotarium, follow the
[Robotarium submission guide](https://www.robotarium.gatech.edu/help)
and submit the same script — no code change required.

## Re-extracting `actor_weights.npy`

The shipped `actor_weights.npy` was produced from
`checkpoint_3000000.pt` (a BenchMARL training checkpoint, ~11 MB, not
included in this repo). If you have that file, regenerate the weights
with:

```bash
python process_checkpoint.py        # reads checkpoint_3000000.pt, writes actor_weights.npy
```

## Dependencies

`torch`, `numpy`, `matplotlib`, `cvxopt` (for the Robotarium barrier
certificates). `rps/` requires Python ≥ 3.5.
