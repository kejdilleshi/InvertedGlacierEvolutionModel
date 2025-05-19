import torch
from data.loader import load_geology
from utils.gpu_monitor import print_gpu_utilization, print_peak_gpu_memory, device
from core.glacier import GlacierDynamicsCheckpointed
from core.inversion import checkpointed_inversion_thicknes
from visualization.plots import plot_gradient_evolution, plot_loss_components

def main():
    # 1) Load data
    Z, obs_thk, mask = load_geology("data/geology_200m.nc")

    # 2) Instantiate model

    rho, g, fd = 910.0, 9.81, 0.25e-16
    Lx, Ly, dx, dy= Z.shape[1]*200, Z.shape[0]*200, 200, 200
    model = GlacierDynamicsCheckpointed(Z, ttot=80, rho=rho, g=g, fd=fd, Lx=Lx, Ly=Ly, dx=dx, dy=dy, dtmax=1.0,
                                        device=device, ice_mask=mask) 

    # 3) Set up optimizer & initial guesses
    precip = torch.tensor(2.3, requires_grad=True, device=device)
    T_ma0 = torch.tensor(4.0, requires_grad=True, device=device)
    T_mj0 = torch.tensor(16.0, requires_grad=True, device=device)

    # # make obs : 
    # H_simulated = model(2.0, 9.0,18.0)
    # torch.save(H_simulated, "Obs_2D.pt")
    optimizer = torch.optim.Adam([precip, T_ma0, T_mj0], lr=0.1)

    # 4) Inversion loop
    grads, losses, data_hist = [], [], []
    for i in range(10):
        optimizer.zero_grad()
        loss, H = checkpointed_inversion_thicknes(precip, T_ma0, T_mj0,
                                      obs_thk.to(device),0,model)
        loss.backward()
        optimizer.step()
        grads.append(torch.norm(precip.grad).item())
        losses.append(loss.item())
        data_hist.append(torch.mean((H - obs_thk)**2).item())
        print(f"Iter {i+1}: loss={loss:.3f}")
    
    # 5) GPU stats & plots
    print_gpu_utilization(); print_peak_gpu_memory()
    plot_gradient_evolution(grads, "grad.png")
    plot_loss_components(losses, data_hist, [], "loss.png")
    print(f" Final precipitation: {precip}, Mean temp {T_ma0} and max {T_mj0}")

if __name__ == "__main__":
    main()
