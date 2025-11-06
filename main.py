import torch
from itertools import product
from data.loader import load_geology
from utils.gpu_monitor import device
from core.glacier import GlacierDynamicsCheckpointed
from core.inversion import inversion_extent, inversion_thicknes
from visualization.plots import plot_loss_components
from config.read_config import parse_arguments  
from core.cnn_model import CNN
import time

def main():
    # === Parse arguments (default config values) ===
    args = parse_arguments()

    # Define the emulator

    config = {
    "nb_layers": 8,               # Number of convolutional layers
    "nb_out_filter": 64,           # Number of output filters for Conv2D
    "conv_ker_size": 5,            # Convolution kernel size
    "activation": "lrelu",          # Activation function: "relu" or "lrelu"
    "dropout_rate": 0.1,           # Dropout rate
    }
    nb_inputs = 3  # thk, slopsurfx, slopsurfy
    nb_outputs = 2  # ubar, vbar
    state = torch.load('data/emulator_model.pth', map_location=device, weights_only=False)
    model = CNN(nb_inputs, nb_outputs, config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()


    # obs_thk = torch.load('Obs_2D.pt').to(device)

    # obs_mask = torch.load('smooth_mask.pt').detach()

    # === Load data once ===
    Z_topo, ice_mask = load_geology("data/geology.nc")

    # Instantiate glacier model (constant across runs)
    Lx, Ly = Z_topo.shape[1] * args.dx, Z_topo.shape[0] * args.dy
    glacier_model = GlacierDynamicsCheckpointed(
        Z_topo, args.ttot, args.rho, args.g, args.fd,
        Lx, Ly, args.dx, args.dy,dtmax=1, device=device, ice_mask=ice_mask,model=model
    )
    # # make observations for P:1.5; T_m:7; T_s: 10 
    _,obs_thk = glacier_model(1.5,7.0, 10.0)
    torch.save(obs_thk,'Obs_2D.pt')
    print('observations generated!')
    exit()
    # === Grid of T and P values ===
    T_values = [8.1]
    P_values = [1.5]

 
    

    for T_init, P_init in product(T_values, P_values):
        # Early stopping parameters
        early_stop_patience = 3
        early_stop_threshold = 1e-5

        # Internal early stopping state
        best_loss = float('inf')
        no_improve_count = 0
        print(f"\n--- Running inversion with T_m_lowest={T_init}, initial_precip={P_init} ---")
        # precip_tensor = torch.full(Z_topo.shape,P_init,requires_grad=True,device=device)
        precip_tensor = torch.tensor(P_init,requires_grad=False,device=device)
        T_m_lowest = torch.tensor(T_init, requires_grad=True, device=device)
        T_s = torch.tensor(10.0, requires_grad=False, device=device)
        optimizer = torch.optim.Adam([T_m_lowest], lr=args.learning_rate)

        loss_hist, Precip_history, data_hist = [], [], []
        results = []
        tag = f"T{T_init}_P{P_init}"

        for i in range(300):  # n_iterations
            optimizer.zero_grad()
            loss, H_simulated, data_fidelity = inversion_extent(
                precip_tensor, T_m_lowest, T_s, obs_thk, glacier_model
            )
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())
            Precip_history.append(T_m_lowest.item())
            data_hist.append(data_fidelity.item())
            print(f"Iter {i+1}: loss={loss:.5f}, extent={data_fidelity:.5f} precip={precip_tensor.mean().item():.3f}, T={T_m_lowest.item():.3f}")
            # Early stopping check
            if i>1:
                if best_loss - loss.item() > early_stop_threshold:
                    best_loss = loss.item()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    print(f"No significant improvement for {no_improve_count} iteration(s).")

                if no_improve_count >= early_stop_patience:
                    print(f"1 Early stopping triggered at iteration {i + 1}.")
                    break
                if loss.item() <= 0.0001:
                    print(f"2 Early stopping triggered at iteration {i + 1}.")
                    break
            
            results.append(f"Iter {i+1}: loss={loss:.5f}, precip={precip_tensor.mean().item():.3f}, T={T_m_lowest.item():.3f}")

        # plot_gradient_evolution(grads, f"grad_{tag}.png")
        plot_loss_components(loss_hist,data_hist, f"data_E/T_only/loss_{tag}.png")
        # torch.save(precip_tensor,f'Recon_Precip_{tag}.pt')
        # torch.save(H_simulated,f'Recon_Glac_{tag}.pt')
        # plot_loss_and_temperature(data_hist,Temp_history=Precip_history,name=f'loss_and_precip_{tag}.png')


        # Save results to file
        with open(f"data_E/T_only/inversion_{tag}.txt", "w") as f:
            for res in results:
                f.write(f"{res}\n")
    print("\n✅ All inversions completed. Results saved in 'inversion_results.txt'.")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")

