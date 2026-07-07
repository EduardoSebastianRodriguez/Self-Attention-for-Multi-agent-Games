#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


from .common import (
    EnsembleModelConfig,
    Model,
    ModelConfig,
    SequenceModel,
    SequenceModelConfig,
)

from .gnn import Gnn, GnnConfig
from .mlp import Mlp, MlpConfig
from .attention import AttentionConfig

classes = [
    "Mlp",
    "MlpConfig",
    "Gnn",
    "GnnConfig",
    "Cnn",
    "CnnConfig",
    "Deepsets",
    "DeepsetsConfig",
    "Gru",
    "GruConfig",
    "Lstm",
    "LstmConfig",
    "SelfAttentionNonlinearPolicy",
    "AttentionConfig"
]

model_config_registry = {
    "mlp": MlpConfig,
    "gnn": GnnConfig,
    "attention": AttentionConfig,
}
