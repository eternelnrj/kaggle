import torch.nn.functional as F
import torch

def petfinder_rmse(input_,target):
    return 100*torch.sqrt(F.mse_loss(torch.sigmoid(input_.flatten()), target))