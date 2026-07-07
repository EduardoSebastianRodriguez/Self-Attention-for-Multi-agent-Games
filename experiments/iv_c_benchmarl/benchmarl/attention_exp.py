"""Sec. IV-C driver: our self-attention policy on Simple-Tag, trained with MAPPO.

Evaluate a trained checkpoint (default) or train from scratch:

    # Evaluate a checkpoint (reproduces the IV-C dist/time metrics)
    python attention_exp.py --checkpoint path/to/checkpoint_3000000.pt

    # Train from scratch (3e6 environment steps)
    python attention_exp.py --train --output outputs/self_attention_simple_tag_mappo
"""
import argparse
from pathlib import Path

import torch_geometric  # noqa: F401  (ensures torch_geometric ops are importable)
from torch import nn

from benchmarl.environments.vmas.common import VmasTask
from benchmarl.algorithms.mappo import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig, AttentionConfig

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "self_attention_simple_tag_mappo"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train from scratch (3e6 steps) instead of evaluating a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pt file to restore when evaluating.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Folder where checkpoints and evaluation results are written.",
    )
    args = parser.parse_args()

    if not args.train and args.checkpoint is None:
        parser.error("evaluation needs --checkpoint (or pass --train to train from scratch)")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig.get_from_yaml()
    config.checkpoint_at_end = True
    config.save_folder = str(output)
    if args.checkpoint is not None:
        config.restore_file = str(Path(args.checkpoint).resolve())

    experiment = Experiment(
        algorithm_config=MappoConfig.get_from_yaml(),
        model_config=AttentionConfig(
            hidden_channels_dim=[64, 64],
            position_key="pos",
            velocity_key="vel",
            exclude_pos_from_node_features=True,
            topology="from_pos",
            self_loops=True,
            edge_radius=1,
            pos_features=2,
            vel_features=2,
        ),
        critic_model_config=MlpConfig(
            num_cells=[256, 256], activation_class=nn.Tanh, layer_class=nn.Linear
        ),
        seed=0,
        config=config,
        task=VmasTask.SIMPLE_TAG_POS.get_from_yaml(),
    )

    if args.train:
        experiment.run()
    else:
        experiment._evaluation_loop(
            get_dist_metrics=True, plot_metrics=True, get_time_metrics=True
        )


if __name__ == "__main__":
    main()
