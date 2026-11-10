import argparse
import torch
import sys

def parse_arguments(arg_list=None):
    parser = argparse.ArgumentParser(description="Parse glacier evolution model parameters.")

    parser.add_argument('--ttot', type=float, default=2017.0, help='Time limit (yr)')
    parser.add_argument('--t_start', type=float, default=1700.0, help='Start time (yr)')
    parser.add_argument('--dtmax', type=float, default=1.0, help='Maximum timestep (yr)')
    parser.add_argument('--cfl', type=float, default=0.2, help='CFL condition for numerical stability')
    parser.add_argument('--initial_mean_temp', type=float, default=7, help='Initial mean temperature (°C)')
    parser.add_argument('--initial_precip', type=float, default=0.2, help='Initial precipitation (m/yr)')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--regularisation', type=float, default=0.05, help='Smoothnes regularization')
    parser.add_argument('--rho', type=float, default=910.0, help='Density of ice (kg/m³)')
    parser.add_argument('--g', type=float, default=9.81, help='Acceleration due to gravity (m/s²)')
    parser.add_argument('--fd', type=float, default=0.25e-16, help='Flow rate factor (Pa⁻³ s⁻¹)')
    parser.add_argument('--dx', type=int, default=100, help='Grid resolution in x direction (m)')
    parser.add_argument('--dy', type=int, default=100, help='Grid resolution in y direction (m)')
    parser.add_argument('--vis_freq', type=float, default=10.0, help='Visualization frequency (yr)')

    # --- Forward scheme and IGM ice-flow emulator ---
    parser.add_argument('--forward_scheme', type=str, default='igm',
                        choices=['SIA', 'igm'],
                        help="Ice dynamics: 'SIA' (shallow ice, internal deformation "
                             "only, no sliding) or 'igm' (IGM emulator, Weertman sliding)")
    parser.add_argument('--emulator_path', type=str,
                        default='emulators/SIADecompNet_18x80_dil1-32',
                        help='IGM emulator artifact directory (manifest.yaml + export/)')
    parser.add_argument('--tau_ref', type=float, default=0.2154,
                        help='Weertman reference basal shear stress (MPa). Lower means '
                             'easier sliding, hence faster flow. Used by the igm scheme')
    parser.add_argument('--arrhenius', type=float, default=78.0,
                        help="Glen's flow-law (Arrhenius) factor, MPa^-3 yr^-1")
    parser.add_argument('--max_velbar', type=float, default=0.0,
                        help="Cap on depth-averaged speed (m/yr); 0 disables. IGM's "
                             'force_max_velbar. Worth setting on DEM-derived geometry, '
                             'where margin cliffs spike the velocity and throttle the CFL step')
    parser.add_argument('--emulator_grad', action='store_true', default=True,
                        help='Backpropagate through the IGM emulator (default)')
    parser.add_argument('--no_emulator_grad', dest='emulator_grad', action='store_false',
                        help='Treat the velocity field as frozen within each step')
    # --- long runs (see GlacierDynamicsCheckpointed) ---
    parser.add_argument('--z_ref', type=float, default=None,
                        help='Elevation (m) the PDD temperature control refers to; '
                             'default: min(ice surface), a moving datum')
    parser.add_argument('--smb_freq', type=float, default=None,
                        help='Years between SMB updates; when set the control is '
                             'interpolated linearly between nodes every vis_freq years')
    parser.add_argument('--chunk_steps', type=int, default=0,
                        help='>0: checkpoint the run in blocks of this many time steps, '
                             'so memory no longer grows with the run length')
    parser.add_argument('--precip_cc', type=float, default=None,
                        help='Clausius-Clapeyron scaling of precipitation with the control '
                             'temperature, fraction per deg C (e.g. 0.07); needs --precip_T_ref')
    parser.add_argument('--precip_T_ref', type=float, default=None,
                        help='Control temperature at which the given precipitation applies')
    parser.add_argument('--outdir', type=str, default='./results/run1', help='Output directory')

    if arg_list is None:
        # Clear Jupyter’s own arguments if running inside a notebook
        argv = sys.argv[:1]
    else:
        argv = arg_list

    args = parser.parse_args(argv[1:])  # skip script name
    # Convert specific arguments to tensors if needed
    args.ttot = torch.tensor(args.ttot)
    args.t_start = torch.tensor(args.t_start)
    return args
