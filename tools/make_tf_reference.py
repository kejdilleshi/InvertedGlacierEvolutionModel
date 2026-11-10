"""Dump a TensorFlow reference forward pass for the SIADecompNet emulator.

The PyTorch port in ``core/emulator/`` has to reproduce IGM's TensorFlow model
exactly. This script runs the original inside an IGM checkout and writes the
inputs, the total velocity and the per-head decomposition to an ``.npz`` that
``tests/test_igm_emulator.py`` compares against.

It is *not* part of the model: it needs TensorFlow and an IGM source tree, so
it is run by hand whenever the emulator artifact changes, and the resulting
``.npz`` is committed.

Usage::

    python tools/make_tf_reference.py \
        --igm /path/to/igm \
        --emulator emulators/SIADecompNet_18x80_dil1-32 \
        --out tests/data/siadecompnet_tf_reference.npz
"""

import argparse
import os
import sys

import h5py
import numpy as np


def collect_named_weights(h5_path):
    """name -> [kernel, bias], read from each layer's stored ``name`` attribute."""
    found = {}
    with h5py.File(h5_path, 'r') as handle:
        def visit(group):
            for key in group:
                item = group[key]
                if not isinstance(item, h5py.Group):
                    continue
                var = item.get('vars')
                if isinstance(var, h5py.Group) and len(var.keys()) > 0:
                    name = var.attrs.get('name')
                    if name:
                        found[str(name)] = [var[str(i)][()] for i in range(len(var.keys()))]
                visit(item)
        visit(handle)
    return found


def conv_layers(model):
    layers = [model.context_in]
    for block in model.context_blocks:
        layers += [block['conv1'], block['conv2']]
    for head in ('slide', 'def', 'res'):
        layers += [getattr(model, f'{head}_head_conv1'),
                   getattr(model, f'{head}_head_conv2'),
                   getattr(model, f'{head}_head_out')]
    return layers


def synthetic_glacier(ny=48, nx=56, seed=0):
    """Must stay in step with ``synthetic_glacier`` in tests/test_igm_emulator.py."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    topg = 3000.0 - 6.0 * yy - 1.5 * xx + 40.0 * np.sin(xx / 7.0) * np.cos(yy / 5.0)
    thk = np.clip(200.0 - 4.0 * ((xx - nx / 2) ** 2 + (yy - ny / 2) ** 2) ** 0.5, 0.0, None)
    thk = np.clip(thk + 5.0 * rng.standard_normal((ny, nx)), 0.0, None)
    return topg, thk


def varying_forcing(ny=48, nx=56):
    """Spatially varying tau_ref / arrhenius / dX.

    A reference built only from uniform forcing cannot detect a swap between
    two constant input channels, so the second case varies all three.
    """
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    tau_ref = 0.08 + 0.4 * (xx / nx)
    arrhenius = 40.0 + 60.0 * (yy / ny)
    dX = np.full((ny, nx), 75.0)
    return tau_ref, arrhenius, dX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--igm', required=True, help='path to an IGM source checkout')
    parser.add_argument('--emulator', required=True, help='emulator artifact directory')
    parser.add_argument('--out', required=True, help='output .npz path')
    parser.add_argument('--tau_ref', type=float, default=0.2)
    parser.add_argument('--arrhenius', type=float, default=78.0)
    parser.add_argument('--dx', type=float, default=100.0)
    args = parser.parse_args()

    sys.path.insert(0, args.igm)
    import tensorflow as tf
    from igm.processes.iceflow.emulate.utils import artifacts as A

    manifest = A.load_supported_manifest(os.path.join(args.emulator, 'manifest.yaml'))
    model = A.Architectures[str(manifest.architecture.name)](**dict(manifest.architecture.params))
    model.input_normalizer = A.build_fixed_input_normalizer_from_manifest(
        manifest, tf.float32, expected_nb_inputs=manifest.nb_inputs, name='input_norm')
    model.build(tf.TensorShape([None, 4, 4, manifest.nb_inputs]))

    # Assign weights by name rather than model.load_weights: Keras only matches
    # the saved object graph, which differs between the versions that wrote
    # these artifacts.
    source = collect_named_weights(os.path.join(args.emulator, 'export', 'weights.weights.h5'))
    for layer in conv_layers(model):
        if layer.name not in source:
            raise SystemExit(f'no stored weights for layer {layer.name!r}')
        current = [tuple(a.shape) for a in layer.get_weights()]
        incoming = [tuple(np.asarray(a).shape) for a in source[layer.name]]
        if current != incoming:
            raise SystemExit(f'shape mismatch for {layer.name}: {incoming} vs {current}')
        layer.set_weights([np.asarray(a) for a in source[layer.name]])

    topg, thk = synthetic_glacier()
    ny, nx = thk.shape

    cases = {
        '': (np.full((ny, nx), args.tau_ref),
             np.full((ny, nx), args.arrhenius),
             np.full((ny, nx), args.dx)),
        'varying_': varying_forcing(ny, nx),
    }

    dump = {}
    for prefix, (tau_ref, arrhenius, dX) in cases.items():
        X = np.stack([thk, topg + thk, tau_ref, arrhenius, dX],
                     axis=-1)[None].astype(np.float32)
        out = model(tf.constant(X), training=False, return_components=True)
        dump[f'ref_{prefix}X'] = X
        dump[f'ref_{prefix}Y'] = np.asarray(out['total'])
        dump[f'ref_{prefix}slide_xy'] = np.asarray(out['slide_xy'])
        dump[f'ref_{prefix}deformation_uv'] = np.asarray(out['deformation_uv'])
        dump[f'ref_{prefix}residual_uv'] = np.asarray(out['residual_uv'])
        if prefix == '':
            for key, value in out['physics_aux'].items():
                dump['ref_aux_' + key] = np.asarray(value)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **dump)
    print(f"wrote {args.out}: Y{dump['ref_Y'].shape} "
          f"range [{dump['ref_Y'].min():.3f}, {dump['ref_Y'].max():.3f}]")


if __name__ == '__main__':
    main()
