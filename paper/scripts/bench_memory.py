"""Peak GPU memory and wall time of one forward + backward pass of the 100 m Argentiere model,
for three ways of storing the computational graph: everything, one checkpoint per step, two-level
checkpointing (blocks of 300 steps). Window shortened to a few decades from 13.6 ka."""
import json, sys, time
import numpy as np, torch
import core.glacier as G
from core.utils import device
from tools.run_argentiere_inversion import CLIMATES, build

mode, years = sys.argv[1], float(sys.argv[2])
full = len(sys.argv) > 3 and sys.argv[3] == "full"
base = CLIMATES["phys"]
chunk = 300 if mode == "twolevel" else 0
glacier, geo, obs, node_times, _ = build("lg", 100, compile_net=False, chunk_steps=chunk, base=base,
                                         valley=True, max_velbar=600.0, adjoint_slope="godunov",
                                         emulator_grad=full)
if mode == "none":                       # keep every intermediate tensor of every step
    G.checkpoint = lambda f, *a, **k: f(*a)
nsteps = [0]
orig = G.GlacierDynamicsCheckpointed._step
def counted(self, *a):
    nsteps[0] += 1
    return orig(self, *a)
G.GlacierDynamicsCheckpointed._step = counted
T0 = np.array(json.load(open("results/argentiere_lg_100m_joint/final_terms.json"))["T"])
T = torch.tensor(T0, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
t_end = float(glacier.t_start) + years
glacier.ttot = torch.tensor(t_end)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
m0 = torch.cuda.memory_allocated()
t0 = time.time()
snaps, H = glacier(precip_tensor=base["precip"], T_m_lowest=T, T_s=base["T_s"],
                   melt_factor=base["melt_factor"], smb_method="PDD", save_times=[t_end - 1e-3])
loss = (H ** 2).mean()
torch.cuda.synchronize(); t1 = time.time()
n_fwd = nsteps[0]
loss.backward()
torch.cuda.synchronize(); t2 = time.time()
peak = torch.cuda.max_memory_allocated() - m0
print(json.dumps(dict(mode=mode, adjoint="full" if full else "frozen", years=years, steps=n_fwd, steps_incl_recompute=nsteps[0],
                      peak_GB=peak / 1e9, fwd_s=t1 - t0, bwd_s=t2 - t1,
                      grad_ok=bool(torch.isfinite(T.grad).all()))), flush=True)
