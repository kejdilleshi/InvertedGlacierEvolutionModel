import numpy as np
import os
import torch
from netCDF4 import Dataset
from scipy.interpolate import interp1d

class Igm_smb_accmelt:

    def __init__(self):
        # Initialize parameters with provided values
        self.thr_temp_rain = 2.5
        self.thr_temp_snow = 0.5
        self.weight_accumulation = 1.0
        self.shift_hydro_year = 0.75
        self.weight_ablation = 1.25
        self.weight_Aletschfirn = 1.0
        self.weight_Jungfraufirn = 1.0
        self.weight_Ewigschneefeld = 1.0
        self.working_dir = "./"  # Default working directory
        self.massbalance_file = "massbalance.nc"
        self.device='cuda:0'

    def init_smb_accmelt(self):
        """
        Load SMB data to run the Aletsch Glacier simulation.
        """
        # Load NetCDF data
        nc = Dataset(os.path.join(self.working_dir, self.massbalance_file), "r")
        x = np.squeeze(nc.variables["x"]).astype("float32")
        y = np.squeeze(nc.variables["y"]).astype("float32")
        self.snow_redistribution = np.squeeze(nc.variables["snow_redistribution"]).astype("float32")
        self.direct_radiation = np.squeeze(nc.variables["direct_radiation"]).astype("float32")
        nc.close()

        if not hasattr(self, "x"):
            self.x = torch.tensor(x, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            assert x.shape == self.x.shape
            assert y.shape == self.y.shape

        # Resample direct radiation at daily resolution
        da = np.arange(20, 351, 30)
        db = np.arange(1, 366)
        self.direct_radiation = interp1d(da, self.direct_radiation, kind="linear", axis=0, fill_value="extrapolate")(db)
        self.direct_radiation = torch.tensor(self.direct_radiation, dtype=torch.float32)

        # Read mass balance parameters
        self.mb_parameters = torch.tensor(
            np.loadtxt(os.path.join(self.working_dir, "mbparameter.dat"), skiprows=2, dtype=np.float32)
        )

        # Expand snow_redistribution to match precipitation (365, ny, nx)
        self.snow_redistribution = torch.unsqueeze(torch.tensor(self.snow_redistribution, dtype=torch.float32), 0)
        self.snow_redistribution = self.snow_redistribution.repeat((365, 1, 1))

        # Define year-corresponding index in mb_parameters
        self.IMB = np.zeros((221), dtype="int32")
        for tt in range(1880, 2101):
            self.IMB[tt - 1880] = np.argwhere(
                (self.mb_parameters[:, 0] <= tt) & (tt < self.mb_parameters[:, 1])
            )[0]
        self.IMB = torch.tensor(self.IMB, dtype=torch.int32)

    def update_smb_accmelt(self):
        """
        Mass balance forced by climate with accumulation and temperature-index melt model [Hock, 1999; Huss et al., 2009].
        """

        # Update melt parameters
        Fm = self.mb_parameters[self.IMB[int(self.t) - 1880], 2] * 10 ** (-3)
        ri = self.mb_parameters[self.IMB[int(self.t) - 1880], 3] * 10 ** (-5)
        rs = self.mb_parameters[self.IMB[int(self.t) - 1880], 4] * 10 ** (-5)

        # Keep solid precipitation when temperature < thr_temp_snow
        accumulation = torch.where(
            self.air_temp <= self.thr_temp_snow,
            self.precipitation,
            torch.where(
                self.air_temp >= self.thr_temp_rain,
                torch.tensor(0.0),
                self.precipitation
                * (self.thr_temp_rain - self.air_temp)
                / (self.thr_temp_rain - self.thr_temp_snow),
            ),
        )
        print(self.precipitation.mean())

        # Unit conversion to daily [m ice eq. / d]
        accumulation /= accumulation.shape[0]
        self.snow_redistribution=self.snow_redistribution.to(self.device)
        # Correct for snow redistribution
        
        
        # self.weight_accumulation=self.weight_accumulation.to(self.device)
        accumulation *= self.snow_redistribution  # [m ice eq. / d]
        accumulation *= self.weight_accumulation  # Apply weight

        pos_temp = torch.where(self.air_temp > 0.0, self.air_temp, torch.tensor(0.0))  # Positive temp [°C]

        ablation = []  # Ablation array [m ice eq. / d]

        # Initialize snow depth
        snow_depth = torch.zeros((self.air_temp.shape[1], self.air_temp.shape[2]),device=self.device)

        # print("self.Fm device:", Fm.device)
        # print("self.ri device:",ri.device)
        self.direct_radiation=self.direct_radiation.to(self.device)

        for kk in range(self.air_temp.shape[0]):
            # Shift to hydro year
            k = (kk + int(self.air_temp.shape[0] * self.shift_hydro_year)) % (self.air_temp.shape[0])

            # Add accumulation to snow depth
            snow_depth += accumulation[k]
            # Calculate ablation
            ablation.append(
                torch.where(
                    snow_depth == 0,
                    pos_temp[k] * (Fm + ri * self.direct_radiation[k]),
                    pos_temp[k] * (Fm + rs * self.direct_radiation[k]),
                )
            )
            ablation[-1] *= self.weight_ablation

            # Update snow depth (ensure non-negative values)
            snow_depth = torch.clamp(snow_depth - ablation[-1], min=0.0)

        # Time integration of accumulation minus ablation
        self.smb = torch.sum(accumulation - torch.stack(ablation), dim=0)

    def load_climate_data_aletsch(self,x_shape,y_shape):
        """
        Load climate data to run the Aletsch Glacier simulation.
        """

        # Altitude of the weather station for climate data
        self.zws = 2766

        # Read temperature and precipitation data from temp_prec.dat
        temp_prec = np.loadtxt(
            os.path.join(self.working_dir, "temp_prec.dat"),
            dtype=np.float32,
            skiprows=2,
        )

        # Find the min and max years from data
        ymin = int(min(temp_prec[:, 0]))
        ymax = int(max(temp_prec[:, 0]))

        self.temp = np.zeros((365, ymax - ymin + 1), dtype=np.float32)
        self.prec = np.zeros((365, ymax - ymin + 1), dtype=np.float32)
        self.year = np.zeros((ymax - ymin + 1), dtype=np.float32)

        # Retrieve temp [unit °C] and prec [unit is m ice eq. / y] and year
        for k, y in enumerate(range(ymin, ymax + 1)):
            IND = (temp_prec[:, 0] == y) & (temp_prec[:, 1] <= 365)
            self.prec[:, k] = (
                temp_prec[IND, -1] * 365.0 / 1000.0
            ) / 0.917  # New unit is m ice eq. / y
            self.temp[:, k] = temp_prec[IND, -2]  # New unit is °C
            self.year[k] = y

        # Initialize air_temp and precipitation fields
        self.air_temp = torch.zeros((len(self.temp), y_shape, x_shape), dtype=torch.float32)
        self.precipitation = torch.zeros((len(self.prec),y_shape, x_shape), dtype=torch.float32)

    def update_climate_aletsch(self,time, x_shape, y_shape,usurf, device,force=False):
        """
        Update climate data for the Aletsch Glacier simulation.
        """

        dP = 0.00035  # Precipitation vertical gradient
        dT = -0.00552  # Temperature vertical gradient

        # Find out the precipitation and temperature at the weather station
        II = self.year == int(time)
        PREC = torch.unsqueeze(torch.unsqueeze(torch.tensor(self.prec[:, II].squeeze(), dtype=torch.float32), dim=-1), dim=-1)
        TEMP = torch.unsqueeze(torch.unsqueeze(torch.tensor(self.temp[:, II].squeeze(), dtype=torch.float32), dim=-1), dim=-1)

        # Extend air_temp and precipitation over the entire glacier and all days of the year
        self.precipitation = PREC.repeat((1, y_shape,x_shape))
        self.air_temp = TEMP.repeat((1, y_shape, x_shape))

        
        # Vertical correction (lapse rates)
        prec_corr_mult = 1 + dP * (usurf - self.zws)
        temp_corr_addi = dT * (usurf - self.zws)
        device = usurf.device if isinstance(usurf, torch.Tensor) else torch.device("cpu")

        # Ensure corrections are on the same device
        prec_corr_mult = prec_corr_mult.to(device)
        temp_corr_addi = temp_corr_addi.to(device)
        self.precipitation=self.precipitation.to(device)
        self.air_temp= self.air_temp.to(device)

        prec_corr_mult = prec_corr_mult.unsqueeze(0).repeat(len(self.prec), 1, 1)
        temp_corr_addi = temp_corr_addi.unsqueeze(0).repeat(len(self.temp), 1, 1)
      
        # The final precipitation and temperature must have shape (365, ny, nx)
        self.precipitation = torch.clamp(self.precipitation * prec_corr_mult, min=0, max=1e10)
        self.air_temp = self.air_temp + temp_corr_addi

# Additional function for surface mass balance
def surface_mass_balance(surface, grad_b=0.01, b_max=1.0, Z_ELA=1500.0):
    # Ensure all inputs are tensors
    if not isinstance(grad_b, torch.Tensor):
        grad_b = torch.tensor(grad_b, dtype=torch.float32)
    if not isinstance(b_max, torch.Tensor):
        b_max = torch.tensor(b_max, dtype=torch.float32)
    if not isinstance(Z_ELA, torch.Tensor):
        Z_ELA = torch.tensor(Z_ELA, dtype=torch.float32)
    # Compute mass balance
    return torch.minimum(grad_b * (surface - Z_ELA), b_max)
