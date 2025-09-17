import torch
import netCDF4
from utils.gpu_monitor import device

def load_geology(path_nc):
    nc = netCDF4.Dataset(path_nc)
    topo = torch.tensor(nc.variables['topg'][:], device=device)
    # thk1880 = torch.tensor(nc.variables['thk_1880'][:], device=device)
    thk1880 = torch.tensor(nc.variables['surf_1880'][:], device=device) -topo
    mask = torch.tensor(nc.variables['icemask'][:], device=device)
    # mask = torch.ones(topo.shape, device=device)
    nc.close()
    return topo,thk1880, mask
