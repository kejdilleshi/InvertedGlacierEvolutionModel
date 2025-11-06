import argparse
import torch
import sys

def parse_arguments(arg_list=None):
    parser = argparse.ArgumentParser(description="Parse glacier evolution model parameters.")

    parser.add_argument('--ttot', type=float, default=2017.0, help='Time limit (yr)')
    parser.add_argument('--t_start', type=float, default=1700.0, help='Start time (yr)')
    parser.add_argument('--dtmax', type=float, default=1.0, help='Maximum timestep (yr)')
    parser.add_argument('--initial_mean_temp', type=float, default=7, help='Initial mean temperature (°C)')
    parser.add_argument('--initial_precip', type=float, default=0.2, help='Initial precipitation (m/yr)')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--regularisation', type=float, default=0.05, help='Smoothnes regularization')
    parser.add_argument('--rho', type=float, default=910.0, help='Density of ice (kg/m³)')
    parser.add_argument('--g', type=float, default=9.81, help='Acceleration due to gravity (m/s²)')
    parser.add_argument('--fd', type=float, default=0.25e-16, help='Flow rate factor (Pa⁻³ s⁻¹)')
    parser.add_argument('--dx', type=int, default=100, help='Grid resolution in x direction (m)')
    parser.add_argument('--dy', type=int, default=100, help='Grid resolution in y direction (m)')
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
