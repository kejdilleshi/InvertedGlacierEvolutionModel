"""Tests for the IGM ice-flow emulator port and its forward scheme.

Two kinds of test live here:

* regression tests that pin behaviour the IGM emulator must not disturb
  (the SIA scheme, config parsing, SMB-method validation);
* correctness tests for the new PyTorch ``SIADecompNet``, its weight
  conversion, the MOLHO column average and the ``igm`` forward scheme.

The reference tensors in ``tests/data/siadecompnet_tf_reference.npz`` were
produced by running the original TensorFlow model in IGM; see
``tools/make_tf_reference.py``.

Run with:  python -m pytest tests/ -v      (or: python tests/test_igm_emulator.py)
"""

import os
import sys
import unittest

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from config.read_config import parse_arguments
from core.emulator import (
    SIADecompNet,
    load_emulator,
    load_manifest,
    vertical_average_weights,
)
from core.glacier import FORWARD_SCHEMES, GlacierDynamicsCheckpointed
from core.inversion import invert_ela

EMULATOR_DIR = os.path.join(REPO, 'emulators', 'SIADecompNet_18x80_dil1-32')
TF_REFERENCE = os.path.join(REPO, 'tests', 'data', 'siadecompnet_tf_reference.npz')

DEVICE = torch.device('cpu')


def synthetic_glacier(ny=48, nx=56, dx=100.0, seed=0):
    """A small, smooth, deterministic ice cap. Same construction as the TF reference."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    topg = 3000.0 - 6.0 * yy - 1.5 * xx + 40.0 * np.sin(xx / 7.0) * np.cos(yy / 5.0)
    thk = np.clip(200.0 - 4.0 * ((xx - nx / 2) ** 2 + (yy - ny / 2) ** 2) ** 0.5, 0.0, None)
    thk = np.clip(thk + 5.0 * rng.standard_normal((ny, nx)), 0.0, None)
    return (torch.tensor(topg, dtype=torch.float32, device=DEVICE),
            torch.tensor(thk, dtype=torch.float32, device=DEVICE))


def base_args(**overrides):
    args = parse_arguments([])
    args.dx = args.dy = 100
    args.dtmax = 1.0
    args.cfl = 0.2
    args.t_start = torch.tensor(0.0)
    args.ttot = torch.tensor(5.0)
    args.vis_freq = 100.0
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# ----------------------------------------------------------------------
# The PyTorch port must reproduce the TensorFlow original
# ----------------------------------------------------------------------
class TestPortMatchesTensorFlow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(TF_REFERENCE):
            raise unittest.SkipTest(f'missing TF reference dump at {TF_REFERENCE}')
        cls.ref = np.load(TF_REFERENCE)
        # Bit-exact IGM parity needs IGM's own slope floor. The default is the
        # larger, gradient-safe SIADecompNet.FIXED_SLOPE_EPS; see its comment.
        cls.emulator = load_emulator(EMULATOR_DIR, device=DEVICE,
                                     slope_eps=SIADecompNet.FIXED_EPS)

    def test_manifest_describes_the_igm_contract(self):
        manifest = load_manifest(EMULATOR_DIR)
        self.assertEqual(manifest.inputs, ['thk', 'usurf', 'slidingco', 'arrhenius', 'dX'])
        # ...but the emulator exposes IGM's 'slidingco' channel under its real name
        emulator = load_emulator(EMULATOR_DIR, device=DEVICE,
                                 slope_eps=SIADecompNet.FIXED_EPS)
        self.assertEqual(emulator.input_names,
                         ['thk', 'usurf', 'tau_ref', 'arrhenius', 'dX'])
        self.assertEqual(manifest.Nz, 2)
        self.assertEqual(manifest.basis_vertical, 'molho')
        self.assertEqual(manifest.nb_outputs, 4)
        self.assertEqual(manifest.output_scale, 1.0)

    def _run(self, prefix):
        X = self.ref[f'ref_{prefix}X']                          # NHWC, from TF
        Xt = torch.from_numpy(np.transpose(X, (0, 3, 1, 2)).copy())
        with torch.no_grad():
            return self.emulator.network(Xt, return_components=True)

    @staticmethod
    def _relative_error(got, tf_nhwc):
        expected = np.transpose(tf_nhwc, (0, 3, 1, 2))
        return np.abs(got.numpy() - expected).max() / max(np.abs(expected).max(), 1e-12)

    def test_forward_pass_matches_tensorflow(self):
        rel = self._relative_error(self._run('')['total'], self.ref['ref_Y'])
        self.assertLess(rel, 1e-5, f'relative error {rel:.3e} vs TensorFlow')

    def test_forward_pass_matches_tensorflow_with_varying_forcing(self):
        """Uniform tau_ref/arrhenius/dX would hide a swap between those channels."""
        rel = self._relative_error(self._run('varying_')['total'], self.ref['ref_varying_Y'])
        self.assertLess(rel, 1e-5, f'relative error {rel:.3e} vs TensorFlow')

    def test_the_two_reference_cases_actually_differ(self):
        """Guards the guard: the varying case must exercise a different response."""
        uniform = self.ref['ref_Y']
        varying = self.ref['ref_varying_Y']
        spread = np.abs(uniform - varying).max() / max(np.abs(uniform).max(), 1e-12)
        self.assertGreater(spread, 0.05)

    def test_head_decomposition_matches_tensorflow(self):
        for prefix in ('', 'varying_'):
            out = self._run(prefix)
            cases = {'slide_xy': self.ref[f'ref_{prefix}slide_xy']}
            for key, tf_key in (('deformation_flat', f'ref_{prefix}deformation_uv'),
                                ('residual_flat', f'ref_{prefix}residual_uv')):
                uv = self.ref[tf_key]                          # [B, H, W, Nz, 2]
                cases[key] = np.concatenate([uv[..., 0], uv[..., 1]], axis=-1)

            for key, tf_value in cases.items():
                rel = self._relative_error(out[key], tf_value)
                self.assertLess(rel, 1e-5, f'{prefix or "uniform "}{key}: rel error {rel:.3e}')

    def test_residual_head_is_not_dead(self):
        """Guards against silently loading zeros into the zero-initialized head."""
        weight = self.emulator.network.res_head_out.weight
        self.assertGreater(float(weight.abs().max()), 0.0)


# ----------------------------------------------------------------------
# Emulator plumbing
# ----------------------------------------------------------------------
class TestEmulatorInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.emulator = load_emulator(EMULATOR_DIR, device=DEVICE)

    def test_molho_weights_are_the_analytic_integrals(self):
        # int phi_bed = 1/(n+2), int phi_surf = (n+1)/(n+2); they must sum to 1
        V_bar = vertical_average_weights('molho', 2, exp_glen=3.0)
        torch.testing.assert_close(V_bar, torch.tensor([0.2, 0.8]))
        self.assertAlmostEqual(float(V_bar.sum()), 1.0, places=6)

    def test_molho_rejects_wrong_nz(self):
        with self.assertRaises(ValueError):
            vertical_average_weights('molho', 3)
        with self.assertRaises(ValueError):
            vertical_average_weights('lagrange', 2)

    def test_column_average_lies_between_the_nodal_velocities(self):
        topg, thk = synthetic_glacier()
        fields = dict(thk=thk, usurf=topg + thk,
                      tau_ref=torch.full_like(thk, 0.2154),
                      arrhenius=torch.full_like(thk, 78.0),
                      dX=torch.full_like(thk, 100.0))
        with torch.no_grad():
            Y = self.emulator(**fields)
            ubar, _ = self.emulator.velocities(**fields)

        ice = thk > 0
        lo = torch.minimum(Y[0], Y[1])[ice]
        hi = torch.maximum(Y[0], Y[1])[ice]
        self.assertTrue(bool(((ubar[ice] >= lo - 1e-4) & (ubar[ice] <= hi + 1e-4)).all()))

    def test_velocities_vanish_off_ice(self):
        topg, thk = synthetic_glacier()
        thk = torch.zeros_like(thk)
        with torch.no_grad():
            ubar, vbar = self.emulator.velocities(
                thk=thk, usurf=topg, tau_ref=torch.full_like(thk, 0.2154),
                arrhenius=torch.full_like(thk, 78.0), dX=torch.full_like(thk, 100.0))
        self.assertEqual(float(ubar.abs().max()), 0.0)
        self.assertEqual(float(vbar.abs().max()), 0.0)

    def test_max_velbar_caps_speed_without_turning_the_flow(self):
        """IGM's force_max_velbar: clip the magnitude, keep the direction."""
        topg, thk = synthetic_glacier()
        fields = dict(thk=thk, usurf=topg + thk,
                      tau_ref=torch.full_like(thk, 0.2154),
                      arrhenius=torch.full_like(thk, 78.0),
                      dX=torch.full_like(thk, 100.0))
        with torch.no_grad():
            ubar, vbar = self.emulator.velocities(**fields)
            cap = float(torch.sqrt(ubar ** 2 + vbar ** 2).max()) / 2.0
            ucap, vcap = self.emulator.velocities(max_velbar=cap, **fields)

        capped_speed = torch.sqrt(ucap ** 2 + vcap ** 2)
        self.assertLessEqual(float(capped_speed.max()), cap * (1 + 1e-5))

        # direction preserved wherever the flow is fast enough to measure
        fast = torch.sqrt(ubar ** 2 + vbar ** 2) > 1.0
        cross = (ubar * vcap - vbar * ucap)[fast].abs().max()
        self.assertLess(float(cross), 1e-3)

        # and pixels already below the cap are untouched
        slow = torch.sqrt(ubar ** 2 + vbar ** 2) < cap * 0.5
        torch.testing.assert_close(ucap[slow], ubar[slow])

    def test_max_velbar_defaults_to_disabled(self):
        self.assertEqual(parse_arguments([]).max_velbar, 0.0)

    def test_missing_input_is_reported_by_name(self):
        topg, thk = synthetic_glacier()
        with self.assertRaises(ValueError) as ctx:
            self.emulator(thk=thk, usurf=topg + thk)
        self.assertIn('tau_ref', str(ctx.exception))

    def test_velocity_responds_to_tau_ref(self):
        """Lower reference shear stress must mean easier sliding, so faster flow."""
        topg, thk = synthetic_glacier()
        speeds = []
        for tau in (0.05, 0.6):
            with torch.no_grad():
                ubar, vbar = self.emulator.velocities(
                    thk=thk, usurf=topg + thk,
                    tau_ref=torch.full_like(thk, tau),
                    arrhenius=torch.full_like(thk, 78.0),
                    dX=torch.full_like(thk, 100.0))
            speeds.append(float(torch.sqrt(ubar**2 + vbar**2)[thk > 1.0].mean()))
        self.assertGreater(speeds[0], speeds[1])

    def test_emulator_weights_are_frozen_by_default(self):
        """The adjoint must not accumulate into the emulator's own parameters."""
        self.assertFalse(any(p.requires_grad for p in self.emulator.parameters()))
        trainable = load_emulator(EMULATOR_DIR, device=DEVICE, trainable=True)
        self.assertTrue(all(p.requires_grad for p in trainable.parameters()))

    def test_frozen_emulator_still_passes_gradient_through(self):
        """Freezing the weights must not cut the path from geometry to velocity."""
        topg, thk = synthetic_glacier()
        thk = thk.clone().requires_grad_(True)
        ubar, vbar = self.emulator.velocities(
            thk=thk, usurf=topg + thk,
            tau_ref=torch.full_like(topg, 0.2154),
            arrhenius=torch.full_like(topg, 78.0),
            dX=torch.full_like(topg, 100.0))
        (ubar.sum() + vbar.sum()).backward()
        self.assertGreater(float(thk.grad.abs().max()), 0.0)
        self.assertIsNone(self.emulator.network.context_in.weight.grad)

    def test_weight_conversion_is_shape_checked(self):
        """A kernel that does not fit the model must raise, not load silently."""
        from core.emulator.keras_weights import load_keras_weights_into
        manifest = load_manifest(EMULATOR_DIR)
        params = dict(manifest.architecture_params)
        params['network_params'] = dict(params['network_params'])
        params['network_params']['nb_out_filter'] = 64          # deliberately wrong
        wrong = SIADecompNet(**params)
        with self.assertRaises(ValueError):
            load_keras_weights_into(wrong, EMULATOR_DIR)


# ----------------------------------------------------------------------
# The 'igm' forward scheme
# ----------------------------------------------------------------------
class TestIgmForwardScheme(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.emulator = load_emulator(EMULATOR_DIR, device=DEVICE)
        cls.topg, cls.thk = synthetic_glacier()

    def build(self, **overrides):
        args = base_args(**overrides)
        return GlacierDynamicsCheckpointed(
            Z_topo=self.topg, H_init=self.thk.clone(),
            ice_mask=torch.ones_like(self.topg), device=DEVICE, args=args,
            emulator=self.emulator, visualize=False,
        )

    def test_runs_and_conserves_plausibility(self):
        model = self.build()
        with torch.no_grad():
            H = model.forward(None, None, None, None, smb_method='field',
                              smb_field=torch.zeros_like(self.topg),
                              forward_scheme='igm')
        self.assertEqual(H.shape, self.topg.shape)
        self.assertTrue(bool(torch.isfinite(H).all()))
        # with zero mass balance, transport must not create ice out of nothing
        self.assertLessEqual(float(H.sum()), float(self.thk.sum()) + 1e-3)

    def test_gradient_flows_to_the_smb_profile(self):
        model = self.build()
        smb_vec = torch.linspace(-8.0, 3.0, 12, device=DEVICE, requires_grad=True)
        H = model.forward(None, None, None, None, smb_method='profile',
                          smb_vec=smb_vec, z_min=self.topg.min().detach(), dz=100.0,
                          forward_scheme='igm')
        H.sum().backward()
        self.assertIsNotNone(smb_vec.grad)
        self.assertTrue(bool(torch.isfinite(smb_vec.grad).all()))
        self.assertGreater(float(smb_vec.grad.abs().max()), 0.0)

    def test_gradient_flows_to_tau_ref_field(self):
        """tau_ref is wired as a field, so it can serve as an inversion control."""
        args = base_args()
        tau = torch.full_like(self.topg, 0.2154).requires_grad_(True)
        model = GlacierDynamicsCheckpointed(
            Z_topo=self.topg, H_init=self.thk.clone(),
            ice_mask=torch.ones_like(self.topg), device=DEVICE, args=args,
            emulator=self.emulator, tau_ref=tau, visualize=False)
        H = model.forward(None, None, None, None, smb_method='field',
                          smb_field=torch.zeros_like(self.topg), forward_scheme='igm')
        H.sum().backward()
        self.assertIsNotNone(tau.grad)
        self.assertGreater(float(tau.grad.abs().max()), 0.0)

    def test_frozen_velocity_mode_still_differentiates_smb(self):
        model = self.build(emulator_grad=False)
        smb = torch.zeros_like(self.topg, requires_grad=True)
        H = model.forward(None, None, None, None, smb_method='field',
                          smb_field=smb, forward_scheme='igm')
        H.sum().backward()
        self.assertGreater(float(smb.grad.abs().max()), 0.0)

    def _volume_fn(self, cfl):
        """Total ice volume after a short run, as a function of the SMB profile."""
        emulator = load_emulator(EMULATOR_DIR, device=DEVICE).double()
        topg, thk = self.topg.double(), self.thk.double()

        args = base_args(cfl=cfl)
        args.ttot = torch.tensor(2.0)
        model = GlacierDynamicsCheckpointed(
            Z_topo=topg, H_init=thk.clone(), ice_mask=torch.ones_like(topg),
            device=DEVICE, args=args, emulator=emulator, visualize=False)

        def volume(smb_vec):
            H = model.forward(None, None, None, None, smb_method='profile',
                              smb_vec=smb_vec, z_min=topg.min().detach(), dz=200.0,
                              forward_scheme='igm')
            return H.sum()

        return volume

    @staticmethod
    def _adjoint_and_fd(volume, smb0, bins, eps=1e-4):
        probe = smb0.clone().requires_grad_(True)
        volume(probe).backward()
        analytic = probe.grad.clone()

        numeric = {}
        for index in bins:
            plus, minus = smb0.clone(), smb0.clone()
            plus[index] += eps
            minus[index] -= eps
            with torch.no_grad():
                numeric[index] = float((volume(plus) - volume(minus)) / (2 * eps))
        return analytic, numeric

    def test_adjoint_matches_finite_differences_at_fixed_timestep(self):
        """The gradient through the emulator must be right, not merely non-zero.

        Run in float64 with the CFL limiter switched off, so every evaluation
        takes the same number of steps of the same length and the finite
        difference measures only what the adjoint claims to compute.
        """
        volume = self._volume_fn(cfl=1e6)      # dt == dtmax every step
        smb0 = torch.linspace(-6.0, 2.0, 8, dtype=torch.float64, device=DEVICE)
        bins = (0, 1, 2, 3)                    # the bins the glacier surface spans
        analytic, numeric = self._adjoint_and_fd(volume, smb0, bins)

        for index in bins:
            scale = max(abs(numeric[index]), 1.0)
            self.assertLess(
                abs(float(analytic[index]) - numeric[index]) / scale, 1e-6,
                f'bin {index}: adjoint {float(analytic[index]):.6g} vs '
                f'finite difference {numeric[index]:.6g}')

    def test_adaptive_timestep_is_the_only_source_of_adjoint_error(self):
        """Document the one approximation in the adjoint.

        ``dt = min(cfl*dx/vel_max, dtmax)`` reads ``vel_max`` through ``.item()``,
        so the time step is a constant as far as autograd is concerned, and
        perturbing the SMB also shifts where the steps land. The legacy
        ``emulator`` scheme makes the same approximation. The adjoint stays the
        right sign and the right order of magnitude; it is not exact.
        """
        volume = self._volume_fn(cfl=0.2)      # CFL-limited, state-dependent dt
        smb0 = torch.linspace(-6.0, 2.0, 8, dtype=torch.float64, device=DEVICE)
        bins = (1, 2)
        analytic, numeric = self._adjoint_and_fd(volume, smb0, bins)

        for index in bins:
            self.assertGreater(float(analytic[index]) * numeric[index], 0.0,
                               f'bin {index}: adjoint and finite difference disagree on sign')
            ratio = float(analytic[index]) / numeric[index]
            self.assertGreater(ratio, 0.5, f'bin {index}: adjoint/fd ratio {ratio:.3f}')
            self.assertLess(ratio, 2.0, f'bin {index}: adjoint/fd ratio {ratio:.3f}')

    def test_missing_emulator_is_a_clear_error(self):
        args = base_args()
        model = GlacierDynamicsCheckpointed(
            Z_topo=self.topg, H_init=self.thk.clone(),
            ice_mask=torch.ones_like(self.topg), device=DEVICE, args=args)
        with self.assertRaises(ValueError) as ctx:
            model.forward(None, None, None, None, smb_method='field',
                          smb_field=torch.zeros_like(self.topg), forward_scheme='igm')
        self.assertIn('emulator', str(ctx.exception))

    def test_non_square_grid_is_rejected(self):
        model = self.build(dy=200)
        with self.assertRaises(ValueError) as ctx:
            model.forward(None, None, None, None, smb_method='field',
                          smb_field=torch.zeros_like(self.topg), forward_scheme='igm')
        self.assertIn('square grid', str(ctx.exception))

    def test_mismatched_forcing_field_is_rejected(self):
        args = base_args()
        with self.assertRaises(ValueError):
            GlacierDynamicsCheckpointed(
                Z_topo=self.topg, H_init=self.thk.clone(),
                ice_mask=torch.ones_like(self.topg), device=DEVICE, args=args,
                emulator=self.emulator, tau_ref=torch.ones(7, 9))


# ----------------------------------------------------------------------
# Regression: everything that worked before must still work
# ----------------------------------------------------------------------
class TestExistingBehaviourUnchanged(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.topg, cls.thk = synthetic_glacier()

    def build(self, **overrides):
        args = base_args(**overrides)
        return GlacierDynamicsCheckpointed(
            Z_topo=self.topg, H_init=self.thk.clone(),
            ice_mask=torch.ones_like(self.topg), device=DEVICE, args=args,
            visualize=False)

    def test_defaults_are_unchanged(self):
        args = parse_arguments([])
        self.assertEqual(args.forward_scheme, 'igm')
        self.assertEqual(args.dx, 100)
        self.assertEqual(args.dy, 100)
        self.assertEqual(args.rho, 910.0)
        self.assertEqual(args.g, 9.81)
        self.assertEqual(args.cfl, 0.2)
        self.assertEqual(args.dtmax, 1.0)
        # new options, with IGM's defaults
        self.assertAlmostEqual(args.tau_ref, 0.2154)
        self.assertAlmostEqual(args.arrhenius, 78.0)
        self.assertTrue(args.emulator_grad)

    def test_glacier_constructs_without_any_emulator(self):
        """SIA needs no emulator, so building without one must still work."""
        model = self.build()
        self.assertIsNone(model.emulator)
        self.assertEqual(model.forward_scheme, 'igm')

    def test_sia_scheme_runs_and_differentiates(self):
        model = self.build()
        smb = torch.zeros_like(self.topg, requires_grad=True)
        H = model.forward(None, None, None, None, smb_method='field',
                          smb_field=smb, forward_scheme='SIA')
        H.sum().backward()
        self.assertTrue(bool(torch.isfinite(H).all()))
        self.assertGreater(float(smb.grad.abs().max()), 0.0)

    def test_unknown_scheme_is_rejected(self):
        model = self.build()
        with self.assertRaises(ValueError):
            model.forward(None, None, None, None, smb_method='field',
                          smb_field=torch.zeros_like(self.topg), forward_scheme='nope')

    def test_smb_methods_still_validate_their_arguments(self):
        model = self.build()
        with self.assertRaises(ValueError):
            model.forward(None, None, None, None, smb_method='ELA', forward_scheme='SIA')
        with self.assertRaises(ValueError):
            model.forward(None, None, None, None, smb_method='field', forward_scheme='SIA')
        with self.assertRaises(ValueError):
            model.forward(None, None, None, None, smb_method='profile', forward_scheme='SIA')
        with self.assertRaises(ValueError):
            model.forward(None, None, None, None, smb_method='bogus', forward_scheme='SIA')


class TestElaInversion(unittest.TestCase):
    """The ELA inversion that IGEM_2D_ELA_igm.ipynb drives."""

    @classmethod
    def setUpClass(cls):
        cls.emulator = load_emulator(EMULATOR_DIR, device=DEVICE)
        cls.topg, cls.thk = synthetic_glacier()

    def build(self):
        args = base_args()
        return GlacierDynamicsCheckpointed(
            Z_topo=self.topg, H_init=self.thk.clone(),
            ice_mask=torch.ones_like(self.topg), device=DEVICE, args=args,
            emulator=self.emulator, visualize=False)

    def observe(self, model, ela):
        with torch.no_grad():
            return model.forward(None, None, None, None, smb_method='ELA',
                                 ELA=torch.as_tensor(ela, device=DEVICE),
                                 grad_b=0.002, b_max=1.5, forward_scheme='igm')

    def test_invert_ela_reports_zero_misfit_at_the_truth(self):
        model = self.build()
        observation = self.observe(model, 2800.0)
        ela = torch.tensor(2800.0, device=DEVICE, requires_grad=True)
        _, loss, data_term, metrics = invert_ela(
            ela, 0.002, 1.5, observation, model)
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertIs(loss, data_term)
        self.assertIn('rmse', metrics)

    def test_invert_ela_gradient_points_downhill(self):
        """A too-high ELA starves the glacier, so lowering it must reduce misfit."""
        model = self.build()
        observation = self.observe(model, 2800.0)

        ela = torch.tensor(3000.0, device=DEVICE, requires_grad=True)
        _, loss_start, _, _ = invert_ela(ela, 0.002, 1.5, observation, model)
        loss_start.backward()
        self.assertGreater(float(loss_start), 0.0)
        self.assertGreater(float(ela.grad), 0.0)      # descent lowers the ELA

        stepped = torch.tensor(float(ela) - 50.0, device=DEVICE, requires_grad=True)
        _, loss_stepped, _, _ = invert_ela(stepped, 0.002, 1.5, observation, model)
        self.assertLess(float(loss_stepped), float(loss_start))

    def test_a_short_inversion_recovers_the_ela(self):
        model = self.build()
        observation = self.observe(model, 2800.0)

        ela = torch.tensor(3000.0, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([ela], lr=25.0)
        for _ in range(12):
            optimizer.zero_grad()
            _, loss, _, _ = invert_ela(ela, 0.002, 1.5, observation, model)
            loss.backward()
            optimizer.step()
        self.assertLess(abs(float(ela) - 2800.0), 200.0 - 1.0,
                        'inversion did not move the ELA towards the truth')


class TestLegacyEmulatorIsGone(unittest.TestCase):
    """The legacy CNN emulator was removed; nothing may quietly depend on it."""

    def test_only_two_forward_schemes_remain(self):
        self.assertEqual(FORWARD_SCHEMES, ('SIA', 'igm'))

    def test_legacy_scheme_name_is_rejected(self):
        args = base_args()
        topg, thk = synthetic_glacier()
        model = GlacierDynamicsCheckpointed(
            Z_topo=topg, H_init=thk, ice_mask=torch.ones_like(topg),
            device=DEVICE, args=args)
        with self.assertRaises(ValueError):
            model.forward(None, None, None, None, smb_method='field',
                          smb_field=torch.zeros_like(topg), forward_scheme='emulator')

    def test_legacy_modules_are_deleted(self):
        for module in ('core.cnn_model', 'models.cnn', 'config.scaling_factors',
                       'core.forward_schemes.emulator_step'):
            with self.assertRaises(ImportError, msg=f'{module} still importable'):
                __import__(module)


if __name__ == '__main__':
    unittest.main(verbosity=2)
