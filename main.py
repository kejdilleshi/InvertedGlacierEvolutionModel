import torch
from data.loader import load_geology
from utils.gpu_monitor import print_gpu_utilization, print_peak_gpu_memory, device
from core.glacier import GlacierDynamicsCheckpointed
from core.inversion import checkpointed_inversion_thicknes
from visualization.plots import plot_gradient_evolution, plot_loss_components

def main():
    # 1) Load data
    Z, obs_thk, mask = load_geology("data/geology_200m.nc")
    T=500
    t=torch.arange(0,T,5)
    P_m=1.7
    P_s=0.5
    P_evol = P_m - P_s * torch.cos(2 * torch.pi * t / T)

    # 2) Instantiate model

    rho, g, fd = 910.0, 9.81, 0.25e-16
    Lx, Ly, dx, dy= Z.shape[1]*200, Z.shape[0]*200, 200, 200
    model = GlacierDynamicsCheckpointed(Z, ttot=T, rho=rho, g=g, fd=fd, Lx=Lx, Ly=Ly, dx=dx, dy=dy, dtmax=1.0,
                                        device=device) 

    # 3) Set up optimizer & initial guesses
    precip = torch.tensor(2.7, requires_grad=False, device=device)
    # T_ma0 = torch.tensor(4.0, requires_grad=True, device=device)
    # T_mj0 = torch.tensor(16.0, requires_grad=True, device=device)

    # # make obs : 
    # H1, H2, H_simulated = model(P_evol, 6,10)
    # print(f'Volume of the glacier at 60 yrs: {torch.sum(H1)}, 80 years : {torch.sum(H2)} and at the end : {torch.sum(H_simulated)}')
    exit()
    # torch.save(H_simulated, "Obs_2D.pt")
    optimizer = torch.optim.Adam([precip], lr=0.1)

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
