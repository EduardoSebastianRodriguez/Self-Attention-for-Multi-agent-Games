#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import Algorithm, AlgorithmConfig
from .ensemble import EnsembleAlgorithm, EnsembleAlgorithmConfig
from .maddpg import Maddpg, MaddpgConfig
from .mappo import Mappo, MappoConfig


classes = [
    "Iddpg",
    "IddpgConfig",
    "Ippo",
    "IppoConfig",
    "Iql",
    "IqlConfig",
    "Isac",
    "IsacConfig",
    "Maddpg",
    "MaddpgConfig",
    "Mappo",
    "MappoConfig",
    "Masac",
    "MasacConfig",
    "Qmix",
    "QmixConfig",
    "Vdn",
    "VdnConfig",
]

# A registry mapping "algoname" to its config dataclass
# This is used to aid loading of algorithms from yaml
algorithm_config_registry = {
    "mappo": MappoConfig,
    "maddpg": MaddpgConfig,
}
