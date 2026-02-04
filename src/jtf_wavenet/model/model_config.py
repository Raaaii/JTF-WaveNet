from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class JTFWaveNetConfig:
    points: int
    filter_count: int
    dilations: Tuple[int, ...]
    blocks: int = 3
    convolution_kernal: Tuple[int, int] = (4, 2)
    initial_factor: float = 1.0
    separate_activation: bool = True
    use_dropout: bool = False
    use_custom_padding: bool = True
    scale_factor_ft: float = 1.0
