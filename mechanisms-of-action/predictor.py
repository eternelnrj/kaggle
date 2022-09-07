import numpy as np

import torch
import torch.nn.functional as F

from dataloaders import DrugsDataset, get_testloader
from loss import convert_predictions
from data_helpers import remove_id_and_fold

class Predictor:
    def __init__(self, fitter, test_df, submission, device="cuda", cp_type=None):
        assert (test_df.sig_id.values == submission.sig_id.values).all()
        
        self.device = device
        self.test_df = test_df
        self.submission = submission
        self.cp_type = cp_type
        
        self.fitter = fitter
        self.fitter.model.to(device)
        
        self.dataset = DrugsDataset(test_df)
        self.dataloader = get_testloader(test_df, batch_size=10)
        
        
    def get_submission(self):
        actions_predictions = []
        submission = self.submission.copy()
        for batch in self.dataloader:
                            
            tabular_features = batch["tabular_features"].to(self.device).float()
            #batch_size = tabular_features.shape[0]
            
            with torch.no_grad():
                outs = convert_predictions(torch.sigmoid(self.fitter.model(tabular_features)), eps=10 ** (-6))  
                
            actions_predictions.append(outs)
                    
        actions_predictions = torch.cat(actions_predictions, dim=0)

        submission.iloc[:, 1:] = actions_predictions.cpu().numpy()
        if self.cp_type is not None:
            selected_columns = set(submission.columns)
            selected_columns.remove("sig_id")
            submission.loc[self.cp_type == "ctl_vehicle", list(selected_columns)]  = 0 #no actions

                   
        return submission
    
    
    def get_average_submission(self, num_folds=5):
        predictions = []
    
        for fold_number in range(num_folds):
            self.fitter.load_best_checkpoint_k_fold(fold_number, num_folds)

            predictions_fold = self.get_submission()
            predictions.append(predictions_fold)
        
        
        average_predictions = self.get_average_submission_from_list(predictions)
    
        return average_predictions
    
    
    def get_average_submission_from_list(self, predictions):
        assert len(predictions) != 0
        
        X = np.zeros_like(remove_id_and_fold(predictions[0]).values)
       
        for p in predictions:
            X += remove_id_and_fold(p).values
        X = X / len(predictions)
       
        average_predictions = predictions[0].copy()
        average_predictions.iloc[:,1:] = X 
    
        return average_predictions