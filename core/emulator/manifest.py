"""Reading of IGM emulator manifests (schema v3).

An IGM emulator artifact is a directory holding

    manifest.yaml
    export/weights.weights.h5

The manifest pins everything needed to rebuild the network bit-for-bit: the
architecture constructor payload, the fixed input-normalization statistics and
the output scale. This module reads it; it deliberately mirrors
``igm/processes/iceflow/emulate/utils/artifacts_schema_v3.py`` so that an
artifact produced by IGM can be consumed here without modification.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


SUPPORTED_SCHEMA_VERSION = 3


@dataclass
class EmulatorManifest:
    schema_version: int
    Nz: int
    basis_vertical: str
    basis_horizontal: str
    inputs: List[str]
    nb_inputs: int
    nb_outputs: int
    output_scale: float
    architecture_name: str
    architecture_params: Dict[str, Any]
    norm_mean: List[float]
    norm_var: List[float]
    norm_epsilon: float


def load_manifest(path) -> EmulatorManifest:
    """Read and validate ``manifest.yaml`` from an emulator directory or file."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing emulator manifest at {path}")

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest at {path} did not parse to a dict, got {type(raw)}")

    schema = int(raw.get("schema_version", -1))
    if schema != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={schema!r}, expected {SUPPORTED_SCHEMA_VERSION}"
        )

    required = {
        "Nz", "basis_vertical", "basis_horizontal", "inputs", "nb_inputs",
        "nb_outputs", "output_scale", "architecture", "normalization",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Manifest {path} is missing required fields: {missing}")

    arch = dict(raw["architecture"])
    norm = dict(raw["normalization"])

    if str(norm.get("method", "")) != "fixed_channel_standardization":
        raise ValueError(
            "Only normalization.method='fixed_channel_standardization' is supported, "
            f"got {norm.get('method')!r}"
        )

    norm_params = dict(norm.get("params", {}))
    for key in ("mean_1d", "var_1d", "epsilon"):
        if key not in norm_params:
            raise ValueError(f"Manifest normalization.params is missing {key!r}")

    inputs = [str(x) for x in raw["inputs"]]
    nb_inputs = int(raw["nb_inputs"])
    if nb_inputs != len(inputs):
        raise ValueError(
            f"Manifest is inconsistent: nb_inputs={nb_inputs} but len(inputs)={len(inputs)}"
        )

    mean = [float(v) for v in norm_params["mean_1d"]]
    var = [float(v) for v in norm_params["var_1d"]]
    if len(mean) != nb_inputs or len(var) != nb_inputs:
        raise ValueError("Normalization statistics length does not match nb_inputs")
    if any(v < 0.0 for v in var):
        raise ValueError("Normalization variance contains negative values")

    arch_params = dict(arch.get("params", {}))
    if [str(x) for x in arch_params.get("input_names", [])] != inputs:
        raise ValueError(
            "Manifest is inconsistent: architecture.params['input_names'] != inputs"
        )
    if int(arch_params.get("Nz", -1)) != int(raw["Nz"]):
        raise ValueError("Manifest is inconsistent: architecture.params['Nz'] != Nz")

    return EmulatorManifest(
        schema_version=schema,
        Nz=int(raw["Nz"]),
        basis_vertical=str(raw["basis_vertical"]),
        basis_horizontal=str(raw["basis_horizontal"]),
        inputs=inputs,
        nb_inputs=nb_inputs,
        nb_outputs=int(raw["nb_outputs"]),
        output_scale=float(raw["output_scale"]),
        architecture_name=str(arch["name"]),
        architecture_params=arch_params,
        norm_mean=mean,
        norm_var=var,
        norm_epsilon=float(norm_params["epsilon"]),
    )
