from jtf_wavenet.model.jtf_wavenet import JTFWaveNet
from jtf_wavenet.model.model_config import JTFWaveNetConfig


def build_jtf_wavenet(cfg: JTFWaveNetConfig) -> JTFWaveNet:
    return JTFWaveNet(
        points=cfg.points,
        filter_count=cfg.filter_count,
        dilations=cfg.dilations,
        blocks=cfg.blocks,
        convolution_kernal=cfg.convolution_kernal,
        initial_factor=cfg.initial_factor,
        separate_activation=cfg.separate_activation,
        use_dropout=cfg.use_dropout,
        use_custom_padding=cfg.use_custom_padding,
        scale_factor_ft=cfg.scale_factor_ft,
    )
