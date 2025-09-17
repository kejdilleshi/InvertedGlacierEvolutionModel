import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse glacier evolution model parameters.")

    parser.add_argument('--ttot', type=int, default=200, 
                        help='Time limit (yr)')
    parser.add_argument('--t_start', type=float, default=1880.0, 
                        help='Start time (yr)')
    parser.add_argument('--initial_mean_temp', type=float, default=7, 
                        help='initial_mean_temperature degree celcius')
    parser.add_argument('--initial_precip', type=float, default=0.2, 
                        help='initial precipitation (m/yr)')
    parser.add_argument('--learning_rate', type=float, default=0.05, 
                        help='learning_rate)')
    parser.add_argument('--rho', type=float, default=910.0, 
                        help='Density of ice (kg/m^3)')
    parser.add_argument('--g', type=float, default=9.81, 
                        help='Acceleration due to gravity (m/s^2)')
    parser.add_argument('--fd', type=float, default=0.25e-16, 
                        help='Flow rate factor (Pa^-3 s^-1)')
    parser.add_argument('--dx', type=int, default=100, 
                        help='Grid resolution in x direction (m)')
    parser.add_argument('--dy', type=int, default=100, 
                        help='Grid resolution in y direction (m)')

    # Filter out Jupyter arguments
    args = parser.parse_args()
    return args
