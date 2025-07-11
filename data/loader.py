import torch
import netCDF4

def load_geology(path_nc):
    nc = netCDF4.Dataset(path_nc)
    topo = torch.tensor(nc.variables['topg'][:], device=torch.device('cuda'))
    thk1880 = torch.tensor(nc.variables['thk_1880'][:], device=torch.device('cuda'))
    mask = torch.tensor(nc.variables['icemask'][:], device=torch.device('cuda'))
    nc.close()
    return topo, thk1880, mask
