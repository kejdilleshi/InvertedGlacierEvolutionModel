
import torch


# # Define a function for the forward computation to use with checkpointing
def checkpointed_inversion_thicknes(Z_ELA,observed_thk, reg_lambda,glacier_model):
    # Perform forward simulation
    H_simulated = glacier_model(Z_ELA)  # Use the checkpointed glacier model
    # print('Nb steps: ',it)

    # Compute data fidelity term
    data_fidelity = torch.mean((H_simulated - observed_thk) ** 2)

    # Compute smoothness regularization
    smoothness_x = torch.sum((Z_ELA[:, 1:] - Z_ELA[:, :-1]) ** 2)
    smoothness_y = torch.sum((Z_ELA[1:, :] - Z_ELA[:-1, :]) ** 2)
    smoothness_reg = smoothness_x + smoothness_y

    # Total loss
    loss = data_fidelity + reg_lambda * smoothness_reg
    return loss,H_simulated

# Define a function for the forward computation to use with checkpointing
def checkpointed_inversion_extent(Z_ELA, observed_thk,glacier_model, threshold=5.0):
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
    H_simulated = glacier_model(Z_ELA)  # Use the checkpointed glacier model
    
    # Convert thickness to binary glacier masks
    mask_simulated = torch.sigmoid((H_simulated - threshold) * 10.0)
    mask_observed = torch.sigmoid((observed_thk - threshold) * 10.0)

    # Compute IoU (Intersection over Union)
    intersection = torch.sum(mask_simulated * mask_observed)  # Element-wise AND
    union = torch.sum(mask_simulated + mask_observed - mask_simulated * mask_observed)  # Element-wise OR
    iou = intersection / (union + 1e-6)  # Add small epsilon for numerical stability

    # Compute IoU loss (1 - IoU)
    iou_loss = 1.0 - iou
    return iou_loss, H_simulated
