"""Register OpenVLA-OFT as an embodied-RL policy.

Set ``embodied.plugin`` to ``rl_plugin``. This directory must be on
``PYTHONPATH``, and LumenRL must be importable. A plugin that also supplies a
world model should import :func:`modeling_openvla_oft.load_openvla_oft` and
:class:`processing_openvla_oft.OpenVLAOFTProcessor` itself.
"""

from lumenrl.core.embodied_plugin import EmbodiedPlugin
from modeling_openvla_oft import load_openvla_oft
from processing_openvla_oft import OpenVLAOFTProcessor

PLUGIN = EmbodiedPlugin(
    build_policy=load_openvla_oft,
    build_processor=OpenVLAOFTProcessor,
)
