import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing


    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class LossMeter:
    def __init__(self):
        self.x = None
        self.y = None
        self.current_loss = 10 ** 9

    
    def reset(self):
        self.x = None
        self.y = None
        self.current_loss = 10 ** 9

        
    def __call__(self, x, y):
        if self.x is None:
            self.x = x
            self.y = y
            
        else:
            self.x = torch.cat([self.x, x], dim=0)
            self.y = torch.cat([self.y, y], dim=0)        
        
        self.current_loss = log_loss(self.x, self.y)
        
    @property
    def avg(self):
        return self.current_loss 
    

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def log_loss(predictions, targets, eps=10 ** (-6)): #predictions.shape = (batch_size, num_input_features), targets.shape = (batch_size, num_input_features)
    assert predictions.shape[0] == targets.shape[0]
    predictions = torch.min(predictions, torch.ones_like(predictions) - eps)
    predictions = torch.max(predictions,  eps * torch.ones_like(predictions))
    
    batch_size = predictions.shape[0]
    num_targets = targets.shape[1]
    
    activation_matrix = targets
    inactivation_matrix = 1 - targets
    
    loss1 = -torch.sum(torch.log(predictions * activation_matrix + inactivation_matrix)) / (batch_size * num_targets)
    loss0 = -torch.sum(torch.log(1  - predictions * inactivation_matrix)) / (batch_size * num_targets)
    
    return loss0 + loss1



def convert_predictions(predictions, eps=10 ** (-6)):
    predictions = torch.min(predictions, torch.ones_like(predictions) - eps)
    predictions = torch.max(predictions,  eps * torch.ones_like(predictions))
    return predictions

