# Sec. IV-C — Pursuit & Evasion in BenchMARL *(placeholder)*

This experiment reproduces **Figs. 6 and 7** of the paper: nonparametric
binomial paired tests over 500 episodes of the Simple Tag pursuit-evasion
game, comparing our attention-based policy against MLP and GAT (graph
attention network) policies trained with **MAPPO** and **MADDPG** in
[BenchMARL](https://github.com/facebookresearch/BenchMARL).

> **Status.** The training code lives in a separate fork that has not
> yet been integrated into this repository. It will be added here once
> our collaborator hands it over.

In the meantime, reviewers who want to reproduce the BenchMARL results
should:

1. Use the modified Simple Tag scenario at
   [`../iv_d_robotarium/simple_tag.py`](../iv_d_robotarium/simple_tag.py)
   (2 teams × 3 holonomic agents, `bound = 1.0`, two obstacles of radius
   0.2, communication radius 1 m).
2. Plug the policy parameterization (`SelfAttentionNonlinearPolicy` in
   [`../iv_d_robotarium/functions.py`](../iv_d_robotarium/functions.py))
   into BenchMARL with the default MAPPO / MADDPG hyperparameters and
   `3 × 10⁶` environment steps.
3. Run the 500-episode head-to-head evaluation described in Sec. IV-C.

A pre-trained MAPPO checkpoint (3 M steps, both teams) that was used to
generate the IV-D Robotarium results is shipped at
[`../iv_d_robotarium/actor_weights.npy`](../iv_d_robotarium/actor_weights.npy).
