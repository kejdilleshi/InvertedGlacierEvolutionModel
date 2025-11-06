# Import the necessary libraries
from core.glacier import GlacierDynamicsCheckpointed
import torch
from core.inversion import inversion_extent,inversion_thicknes
from visualization.plots import  plot_loss_components,plot_temp, plot_sim_obs_extents
from core.cnn_model import CNN
from data.loader import load_geology
from core.utils import metrics_to_str,iterations_to_jsonl
import netCDF4
from config.read_config import parse_arguments  
import wandb
import os


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
args = parse_arguments()

# after args = parse_arguments()
wandb.init(project="IGEM")
# override from sweep
args.learning_rate = wandb.config.get("learning_rate", args.learning_rate)
args.regularisation = wandb.config.get("regularisation", args.regularisation)
args.outdir = f"./results/ext_mean_l2_1926_seq_7/{wandb.run.name}"
os.makedirs(args.outdir, exist_ok=True)


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

Z_topo,H_init, icemask = load_geology("data/geology.nc")

dx=100
dy=100
args.ttot=torch.tensor(2017.0)
args.t_start=torch.tensor(1700.0)
rho, g, fd=1,1,1
precip_tensor=1.85
T_s=10.0


nc = netCDF4.Dataset('./data/geology.nc')
topo = torch.tensor(nc.variables['topg'][:], device=device)
obs_1880= torch.tensor(nc.variables['surf_1880'][:], device=device) -topo
obs_26= torch.tensor(nc.variables['surf_1926'][:], device=device) -topo
obs_57= torch.tensor(nc.variables['surf_1957'][:], device=device) -topo
obs_80= torch.tensor(nc.variables['surf_1980'][:], device=device) -topo
obs_99= torch.tensor(nc.variables['surf_1999'][:], device=device) -topo
obs_09= torch.tensor(nc.variables['surf_2009'][:], device=device) -topo
obs_17= torch.tensor(nc.variables['surf_2017'][:], device=device) -topo

nc.close()

glacier_model = GlacierDynamicsCheckpointed(
        Z_topo,H_init, ice_mask=icemask,device=device,args=args,model=model
    )
m_f_list=[2.7/12]

for melt_factor in m_f_list:
    #update the location of the output
    T_m_lowest = torch.full((int((args.ttot-args.t_start)/10 +1),1),7.0, requires_grad=True, device=device)
    # # Create frozen and trainable parts
    # T_frozen = torch.full((int((1850-args.t_start)/10), 1), 7.0, requires_grad=False, device=device)
    # T_trainable = torch.full((int((args.ttot-1850)/10 +1),1), 7.0, requires_grad=True, device=device)    


    P_daily_=None
    T_daily_=None

    optimizer = torch.optim.Adam([T_m_lowest], lr=args.learning_rate)
    loss_hist, logs, data_hist = [], [], []
    results = []
    #define the weights of loss values:
    w1880=1.0
    w26=1.0
    w57=0.0
    w80=0.0
    w99=0.0
    w09=0.0
    w17=0.0

    for i in range(110):  # n_iterations
        optimizer.zero_grad()

        # Force positivity with softplus
        if i >20: 
            w57=1.0
        if i > 40: 
            w80=1.0
        if i > 60:
            w99=1.0
            w09=1.0
        if i > 80: 
            w17=1.0
        # T_m_lowest = torch.cat([T_frozen, T_trainable], dim=0)

        H_simulated, loss,data, metrics = inversion_extent(precip_tensor,T_m_lowest,T_s,
                                                        P_daily=P_daily_,T_daily=T_daily_,
                                                        melt_factor=melt_factor,
                                                        obs1880=obs_1880, obs26=obs_26 ,obs57=obs_57 ,obs80=obs_80 ,obs99=obs_99 ,obs09=obs_09, obs17= obs_17,
                                                        glacier_model=glacier_model,
                                                        reg_lambda=args.regularisation,
                                                        w1880=w1880, w26=w26,w57=w57,w80=w80,w99=w99,w09=w09,w17=w17)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        data_hist.append(data.item())
        logs.append(f"loss={loss.item():.5f}, metrics = {metrics_to_str(metrics)}")
        wandb.log({"loss/total": loss.item(),"loss/data": data.item(),"grad/mean": T_m_lowest.grad.mean().item(),"grad/min": T_m_lowest.grad.min().item(),"grad/max": T_m_lowest.grad.max().item(),
                        "Temperature": T_m_lowest,
                    })

        if i%5==0:
            plot_temp(T_m_lowest, i,args)
            plot_sim_obs_extents(H_simulated,[obs_1880, obs_26 ,obs_57 ,obs_80 ,obs_99 ,obs_09, obs_17],args,iter=i)
            # plot_sim_with_obs_extent(sim=H_simulated, obs=obs_17,year=2017, thresh=1.0, iter= i))
        # Precip_history.append(T_m_lowest.item())
        # print(f"Iter {i+1}: loss={loss:.5f}, extent={data_fidelity:.5f} precip={precip_tensor.mean().item():.3f}, T={T_m_lowest.mean().item():.3f}")
        print(f"Iter {i+1}: loss={loss.item():.5f}, data={data.item():.5f}, "
              f"gradient mean={T_m_lowest.grad.mean().item():.5f}, "
              f"min={T_m_lowest.grad.min().item():.5f}, "
              f"max={T_m_lowest.grad.max().item():.5f}")
    plot_temp(T_m_lowest, i,args)
    plot_loss_components(loss_hist,data_hist,args) 
    iterations_to_jsonl(logs,args)
    # Save args as JSON
    # with open(os.path.join(args.outdir, "args.json"), "w") as f:
    #     json.dump(vars(args), f, indent=4)


