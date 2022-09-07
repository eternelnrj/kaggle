import pandas as pd
from torch.optim import Adam
from path import Path

from fitter import Fitter
from predictor import Predictor

from config import config
from seed import set_seed
from processing_data import create_df_and_targets
from loss import SmoothBCEwLogits
from data_helpers import get_cp_type_col


class TrainConfig:
    num_workers = 4
    batch_size = 128
    n_epochs = 1
    lr = 0.0005
    weight_decay = 0.00001

    num_folds = 5
    
    model_name = "three_layers_model"
    num_input_features = 1096
    
    device = "cuda"
    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    criterion_class = SmoothBCEwLogits
    criterion_params = {"smoothing" : 0.001}
    
    step_scheduler = True  # do scheduler.step after optimizer.step
    validation_scheduler = False  # do scheduler.step after validation stage loss
    
    optimizer_class = Adam
    optimizer_params = {"lr" : lr, "weight_decay" : weight_decay}
    


set_seed()
print("Creating dataframes...")
full_df, train_targets = create_df_and_targets(TrainConfig)

train_df = full_df[full_df.WHERE=="train"].reset_index(drop=True)
test_df = full_df[full_df.WHERE=="test"].reset_index(drop=True)

train_df =  train_df.drop("WHERE", axis=1)
test_df = test_df.drop(["WHERE", "fold"], axis=1)

fitter = Fitter(TrainConfig)
fitter.train_k_fold(train_df, train_targets, total_folds=TrainConfig.num_folds, continue_from_checkpoint=False)


sample_submission = pd.read_csv(Path(config["path_to_data_dir"]) / "sample_submission.csv")
predictor = Predictor(fitter, test_df, sample_submission, device="cpu", cp_type=get_cp_type_col())
submission = predictor.get_average_submission(num_folds=TrainConfig.num_folds)
submission.to_csv("submission.csv", index=False)