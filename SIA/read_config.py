import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse glacier evolution model parameters.")

    parser.add_argument('--ttot', type=int, default=1900, 
                        help='Time limit (yr)')
    parser.add_argument('--t_start', type=float, default=1880.0, 
                        help='Start time (yr)')
    parser.add_argument('--grad_b', type=float, default=0.001, 
                        help='Mass balance gradient')
    parser.add_argument('--b_max', type=float, default=1.0, 
                        help='Maximum precip (m/yr)')
    parser.add_argument('--Z_ELA', type=float, default=3000.0, 
                        help='Elevation of equilibrium line altitude (m)')
    parser.add_argument('--rho', type=float, default=910.0, 
                        help='Density of ice (kg/m^3)')
    parser.add_argument('--g', type=float, default=9.81, 
                        help='Acceleration due to gravity (m/s^2)')
    parser.add_argument('--fd', type=float, default=1.9e-24, 
                        help='Flow rate factor (Pa^-3 s^-1)')
    parser.add_argument('--dx', type=int, default=100, 
                        help='Grid resolution in x direction (m)')
    parser.add_argument('--dy', type=int, default=100, 
                        help='Grid resolution in y direction (m)')

    # Filter out Jupyter arguments
    args = parser.parse_args()
    return args
