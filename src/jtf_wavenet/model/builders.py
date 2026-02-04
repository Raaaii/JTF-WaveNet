from __future__ import annotations
from jtf_wavenet.model.jtf_wavenet import JTFWaveNet, JTFWaveNetConfig


from jtf_wavenet.model.jtf_wavenet import JTFWaveNet, JTFWaveNetConfig


def build_jtfwavenet(cfg: JTFWaveNetConfig):
    return JTFWaveNet(cfg=cfg)
