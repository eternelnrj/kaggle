from fastai.vision.all import *
import torch
import numpy as np
from timm import create_model
import gc

from processing_data import get_train_test_dfs, get_dataloaders
from metrics import petfinder_rmse


train_config = {"num_folds" :  5, "num_workers" : 6, "model_name" : "swin_large_patch4_window12_384"}
train_df, test_df = get_train_test_dfs(train_config)


for fold in range(train_config["num_folds"]):
    model = create_model(train_config["model_name"], pretrained=True, num_classes=1)
    
    dataloaders = get_dataloaders(train_df, fold, train_config)
    learner = Learner(dataloaders, model, loss_func=BCEWithLogitsLossFlat(), metrics=petfinder_rmse, lr=10**(-6),
                    cbs=[SaveModelCallback(with_opt=True, fname=f"ckpt_fold_{fold}"),
                    EarlyStoppingCallback(monitor='petfinder_rmse', comp=np.less, patience=3)])
    learner.fit_one_cycle(20)
    
    del learner
    torch.cuda.empty_cache()
    gc.collect()


test_dl = dataloaders.test_dl(test_df)
predictions = []

for fold in range(train_config["num_folds"]):
    model = create_model(train_config["model_name"], pretrained=False, num_classes=1)
    load_model(f"./models/ckpt_fold_{fold}.pth", model, opt=None, with_opt=False, device=torch.device("cuda"), strict=True)
    learner = Learner(dataloaders, model, loss_func=BCEWithLogitsLossFlat(), metrics=petfinder_rmse, lr=10**(-6))

    predictions_, _ = learner.tta(dl=test_dl, n=5)
    predictions.append(predictions_)
    

predictions = np.stack(predictions).mean(axis=0)
test_df["Pawpularity"] = (100 * predictions).squeeze()
submission = test_df[["Id","Pawpularity"]]
submission.to_csv("submission.csv", index=False)