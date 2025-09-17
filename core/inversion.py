from torch.utils.checkpoint import checkpoint
import torch
import torch.nn.functional as Fnn



# # Define a function for the forward computation to use with checkpointing
# def inversion_thicknes(precip_tensor, T_m_lowest,T_s,observed_thk, glacier_model,reg_lambda=0.0001):
#     # Perform forward simulation
#     H_simulated = checkpoint(glacier_model,precip_tensor,T_m_lowest, T_s)
#     # Compute data fidelity term
#     data_fidelity = torch.mean(torch.abs(H_simulated - observed_thk) ** 2)/torch.norm(observed_thk)

#     # Compute smoothness regularization
#     smoothness_x = torch.sum((precip_tensor[:, 1:] - precip_tensor[:, :-1]) ** 2)
#     smoothness_y = torch.sum((precip_tensor[1:, :] - precip_tensor[:-1, :]) ** 2)
#     smoothness_reg = smoothness_x + smoothness_y

#     # Total loss
#     loss = data_fidelity + reg_lambda * smoothness_reg
#     return loss,H_simulated, data_fidelity

#version for daily temperature and prec
def inversion_thicknes(P_daily, T_daily,melt_factor,obs26, obs57, obs80,obs99, glacier_model,reg_lambda=0.0001):
    # Perform forward simulation

    # ---- run model ----
    H26, H57, H80, H99 = glacier_model(
        precip_tensor=0, T_m_lowest=0, T_s=0,
        P_daily=P_daily, T_daily=T_daily, melt_factor=melt_factor
    )
    print()
    
    
    # ---- mask both sim and obs (soft threshold) ----
    thresh = 10.0      # meters
    scale  = 100.0    # steeper -> closer to hard 0/1
    
    H26_m  = torch.sigmoid(scale * (H26  - thresh))
    H57_m  = torch.sigmoid(scale * (H57  - thresh))
    H80_m  = torch.sigmoid(scale * (H80  - thresh))
    H99_m  = torch.sigmoid(scale * (H99  - thresh))
    
    obs26_m = torch.sigmoid(scale * (obs26 - thresh))
    obs57_m = torch.sigmoid(scale * (obs57 - thresh))
    obs80_m = torch.sigmoid(scale * (obs80 - thresh))
    obs99_m = torch.sigmoid(scale * (obs99 - thresh))
    
    # ---- MAS/MSE on masks ----
    data_fidelity1 = torch.mean((H26 - obs26) ** 2)
    data_fidelity2 = torch.mean((H57 - obs57) ** 2)
    data_fidelity3 = torch.mean((H80 - obs80) ** 2)
    data_fidelity4 = torch.mean((H99 - obs99) ** 2)
    
    loss =   data_fidelity4 + data_fidelity3 + data_fidelity2+ data_fidelity1



    # Total loss
    return H99, loss






# def inversion_thicknes(precip_tensor, T_m_lowest, T_s, observed_thk, glacier_model, mask, reg_lambda=0.00001):
#     # Perform forward simulation
#     H_simulated = checkpoint(glacier_model, precip_tensor, T_m_lowest, T_s)

#     # Masked observations: original where mask==1, else 1
#     masked_obs = observed_thk * mask 

#     # Masked simulated thickness: original where mask==1, else 1
#     masked_sim = H_simulated * mask 

#     # Steeper sigmoid using a scaling factor
#     scale = 100.0
#     ext_simulated = torch.sigmoid(scale * (H_simulated - 1))
#     ext_observed = torch.sigmoid(scale * (observed_thk - 1))

#     # Compute data fidelity term
#     extent_fidelity = torch.mean(torch.abs(ext_simulated - ext_observed) ** 2)

#     thk_fidelity= torch.mean(torch.abs(masked_sim - masked_obs) ** 2) / torch.norm(masked_obs)

#     # Compute data fidelity term (using masked values)
#     # data_fidelity = torch.mean(torch.abs(masked_sim - masked_obs) ** 2) / torch.norm(masked_obs)

#     # Compute smoothness regularization on precipitation
#     smoothness_x = torch.sum((precip_tensor[:, 1:] - precip_tensor[:, :-1]) ** 2)
#     smoothness_y = torch.sum((precip_tensor[1:, :] - precip_tensor[:-1, :]) ** 2)
#     smoothness_reg = smoothness_x + smoothness_y

#     # Total loss
#     loss = extent_fidelity+ 0.1* thk_fidelity + reg_lambda * smoothness_reg

#     return loss, H_simulated, extent_fidelity, 0.1* thk_fidelity, reg_lambda * smoothness_reg



# Define a function for the forward computation to use with checkpointing
# def inversion_extent(precip_tensor, T_m_lowest, T_s,observed_thk,glacier_model,reg_lambda=0.001):
#     """
#     Forward computation with IoU for data fidelity and smoothness regularization.
#     Args:
#         Z_ELA (torch.Tensor): The equilibrium line altitude field.
#         observed_thk (torch.Tensor): The observed glacier thickness.
#         reg_lambda (float): Regularization parameter for smoothness.
#         threshold (float): Thickness threshold to define glacier extent.
#     Returns:
#         loss (torch.Tensor): Total loss including IoU and regularization.
#         H_simulated (torch.Tensor): Simulated glacier thickness.
#     """
#     # Perform forward simulation
#     H_simulated = checkpoint(glacier_model,precip_tensor,T_m_lowest, T_s)  # Use the checkpointed glacier model
    
#     # Steeper sigmoid using a scaling factor
#     scale = 100.0
#     mask_simulated = torch.sigmoid(scale * (H_simulated - 1))
#     mask_observed = torch.sigmoid(scale * (observed_thk - 1))

#     # Compute data fidelity term
#     data_fidelity = torch.mean(torch.abs(mask_simulated - mask_observed) ** 2)

#     # # Compute smoothness regularization
#     # smoothness_x = torch.sum((precip_tensor[:, 1:] - precip_tensor[:, :-1]) ** 2)
#     # smoothness_y = torch.sum((precip_tensor[1:, :] - precip_tensor[:-1, :]) ** 2)
#     # smoothness_reg = smoothness_x + smoothness_y
    
#     # Total loss
#     loss = data_fidelity #+ reg_lambda * smoothness_reg

#     return loss, H_simulated, data_fidelity

def inversion_extent(precip_tensor, T_m_lowest, T_s,observed_thk1,observed_thk2,glacier_model,reg_lambda=0.001):
    """
    Forward computation with IoU for data fidelity and smoothness regularization.
    Args:
        Z_ELA (torch.Tensor): The equilibrium line altitude field.
        observed_thk (torch.Tensor): The observed glacier thickness.
        reg_lambda (float): Regularization parameter for smoothness.
        threshold (float): Thickness threshold to define glacier extent.
    Returns:
        loss (torch.Tensor): Total loss including IoU and regularization.
        H_simulated (torch.Tensor): Simulated glacier thickness.
    """
    # Perform forward simulation
    H_simulated1, H_simulated2 = checkpoint(glacier_model,precip_tensor,T_m_lowest, T_s)  # Use the checkpointed glacier model
    
    # Steeper sigmoid using a scaling factor
    scale = 100.0
    mask_simulated1 = torch.sigmoid(scale * (H_simulated1 - 1))
    mask_simulated2 = torch.sigmoid(scale * (H_simulated2 - 1))

    mask_observed1 = torch.sigmoid(scale * (observed_thk1 - 1))
    mask_observed2 = torch.sigmoid(scale * (observed_thk2 - 1))


    # Compute data fidelity term
    data_fidelity1 = torch.mean(torch.abs(mask_simulated1 - mask_observed1) ** 2)
    data_fidelity2 = torch.mean(torch.abs(mask_simulated2 - mask_observed2) ** 2)


    # # Compute smoothness regularization
    smoothness_x = torch.sum((T_m_lowest[1:] - T_m_lowest[:-1]) ** 2)
    # smoothness_y = torch.sum((precip_tensor[1:, :] - precip_tensor[:-1, :]) ** 2)
    # smoothness_reg = smoothness_x + smoothness_y
    
    # Total loss
    loss =  data_fidelity1 + data_fidelity2 + reg_lambda* smoothness_x

    return loss, H_simulated1, reg_lambda*smoothness_x

