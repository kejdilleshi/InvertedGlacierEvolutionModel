import torch
import netCDF4
from utils.gpu_monitor import device

def load_geology(path_nc):
    nc = netCDF4.Dataset(path_nc)
    topo = torch.tensor(nc.variables['topg'][:], device=device)
    # thk1880 = torch.tensor(nc.variables['thk_1880'][:], device=device)
    thk1880 = torch.tensor(nc.variables['surf_1999'][:], device=device) -topo
    mask = torch.tensor(nc.variables['icemask'][:], device=device)
    # mask = torch.ones(topo.shape, device=device)
    nc.close()
    return topo,thk1880, mask

def load_daily_data(file_path, accumulate=False, device='cpu'):
    """
    Parse a whitespace-delimited file with columns:
        year  jd  hour  temp  prec
    (first two lines are headers), and pack into a tensor of shape:
        (num_years, 366, 2), where [:, :, 0] = temp, [:, :, 1] = prec

    Parameters
    ----------
    file_path : str
        Path to the input file (e.g., "temp_prec.dat").
    accumulate : bool, optional
        If multiple rows map to the same (year, day), whether to sum them (True)
        or let the last occurrence overwrite (False). Default False.
    device : str or torch.device, optional
        Device to place the resulting tensors on.

    Returns
    -------
    data : torch.FloatTensor
        Shape (num_years, 366, 2): temperature and precipitation grids.
    years : torch.LongTensor
        Shape (num_years,): sorted unique calendar years aligned with data[year_idx].
    """
    # --- Read & parse file without NumPy/pandas ---
    years_f, jds_f, temps, precs = [], [], [], []

    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()

    # Skip first 2 header rows
    for line in lines[2:]:
        if not line.strip():
            continue
        cols = line.split()
        if len(cols) < 5:
            continue
        y, jd, hr, te, pr = cols[:5]
        years_f.append(float(y))
        jds_f.append(float(jd))
        temps.append(float(te))
        precs.append(float(pr))

    if not years_f:
        # No data rows parsed; return empty tensors
        empty = torch.zeros((0, 366, 2), dtype=torch.float32, device=device)
        return empty, torch.zeros((0,), dtype=torch.int64, device=device)

    # --- Convert to tensors ---
    year = torch.tensor(years_f, dtype=torch.float32, device=device)
    jd   = torch.tensor(jds_f,   dtype=torch.float32, device=device)
    temp = torch.tensor(temps,   dtype=torch.float32, device=device)
    prec = torch.tensor(precs,   dtype=torch.float32, device=device)

    # Cast year & jd to int
    year_i = year.to(torch.int64)
    jd_i   = jd.to(torch.int64)

    # Optional: sort by (year, jd) for tidiness
    sort_keys = year_i * 1000 + jd_i
    order = torch.argsort(sort_keys)
    year_i = year_i[order]
    jd_i   = jd_i[order]
    temp   = temp[order]
    prec   = prec[order]

    # Unique sorted years and mapping (yidx)
    years = torch.unique(year_i, sorted=True)
    yidx  = torch.searchsorted(years, year_i)

    # Day index: clamp to 1..366, then make 0-based
    didx = torch.clamp(jd_i, 1, 366) - 1

    num_years = years.numel()

    # --- Scatter into grids ---
    temp_grid = torch.zeros((num_years, 366), dtype=torch.float32, device=device)
    prec_grid = torch.zeros((num_years, 366), dtype=torch.float32, device=device)

    # If you have multiple rows per (year, day) and want to SUM them, set accumulate=True
    temp_grid.index_put_((yidx, didx), temp, accumulate=accumulate)
    prec_grid.index_put_((yidx, didx), prec, accumulate=accumulate)

    # Stack to (num_years, 366, 2)
    data = torch.stack((temp_grid, prec_grid), dim=-1)

    return data, years

