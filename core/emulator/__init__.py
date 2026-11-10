"""IGM ice-flow emulator, ported to PyTorch.

Loads an IGM emulator artifact (``manifest.yaml`` + ``export/weights.weights.h5``)
into a differentiable ``torch.nn.Module`` so that gradients flow from the
simulated ice thickness back to whatever is being inverted for.
"""

from core.emulator.manifest import EmulatorManifest, load_manifest
from core.emulator.siadecompnet import FixedChannelStandardization, SIADecompNet
from core.emulator.keras_weights import load_keras_weights_into
from core.emulator.loader import IceFlowEmulator, load_emulator, vertical_average_weights

__all__ = [
    "EmulatorManifest",
    "FixedChannelStandardization",
    "IceFlowEmulator",
    "SIADecompNet",
    "load_emulator",
    "load_keras_weights_into",
    "load_manifest",
    "vertical_average_weights",
]
