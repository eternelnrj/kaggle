import torch
from torch import nn
import torch.nn.functional as F

class Mask(nn.Module):
    def __init__(self, num_input_features=1096, num_inner_features=2048):
        super(Mask, self).__init__()
        
        self.fcl_input = nn.Sequential(nn.Linear(num_input_features, num_inner_features),\
                                        nn.ReLU())
        
        self.fcl_out = nn.Sequential(nn.BatchNorm1d(num_inner_features), \
                                     nn.Linear(num_inner_features, num_input_features), nn.Sigmoid())
        
    def forward(self, out):
        out = self.fcl_input(out)
        out = self.fcl_out(out)
        
        return out
    
class ThreeLayersNetwork(nn.Module):
    def __init__(self, num_input_features=1096, num_inner_features=2048, num_output_features=206):
        super(ThreeLayersNetwork, self).__init__()
        self.mask = Mask(num_input_features) 
        
        self.batch_norm0 = nn.BatchNorm1d(num_input_features)
        
        self.fcl_input = nn.utils.weight_norm(nn.Linear(num_input_features, num_inner_features))
        self.batch_norm1 = nn.BatchNorm1d(num_inner_features)
        
        
        self.fcl_inner =  nn.utils.weight_norm(nn.Linear(num_inner_features, num_inner_features))
        self.batch_norm2 = nn.BatchNorm1d(num_inner_features)
        
        
        self.fcl_output = nn.utils.weight_norm(nn.Linear(num_inner_features, num_output_features))
        
        self.dropout0 = nn.Dropout(p=0.1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        
        
    def forward(self, out):
        out = self.mask(out) * out
        
        out = self.batch_norm0(out)
        out = self.dropout0(out)
        
        self.recalibrate_layer(self.fcl_input)
        out = self.dropout1(self.batch_norm1(torch.relu(self.fcl_input(out))))
        
        self.recalibrate_layer(self.fcl_inner)
        out = self.dropout2(self.batch_norm2(torch.relu(self.fcl_inner(out))))

        self.recalibrate_layer(self.fcl_output)
        out = self.fcl_output(out)
        
        return out
    
    
    def recalibrate_layer(self, layer):
        if(torch.isnan(layer.weight_v).sum() > 0):
            print('recalibrate layer.weight_v')
            layer.weight_v = torch.where(torch.isnan(layer.weight_v), torch.zeros_like(layer.weight_v) + 1e-7, layer.weight_v)
            
            
        if(torch.isnan(layer.weight).sum() > 0):
            print('recalibrate layer.weight')
            layer.weight = torch.where(torch.isnan(layer.weight), torch.zeros_like(layer.weight) + 1e-7, layer.weight)
            
    
def make_model(config):
    if config.model_name=="three_layers_model":
        return ThreeLayersNetwork(num_input_features=config.num_input_features, num_inner_features=2048, num_output_features=206)