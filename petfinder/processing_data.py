import pandas as pd
from config import config
from path import Path
from sklearn.model_selection import StratifiedKFold
from fastai.vision.all import *



def get_train_test_dfs(train_config):
    path_data = Path(config["path_to_data_dir"])
    path_train_df = path_data / "train.csv"
    path_test_df = path_data / "test.csv"

    train_df = pd.read_csv(path_train_df)
    test_df = pd.read_csv(path_test_df)

    print(path_data)
    train_df["path"] = train_df["Id"].apply(lambda x : path_data  / "train" / (x + ".jpg"))
    test_df["path"] = test_df["Id"].apply(lambda x : path_data  / "test" / (x + ".jpg"))
    
    train_df["score"] = train_df["Pawpularity"].apply(lambda x : x / 100)
    train_df["bucket"] = train_df["Pawpularity"].apply(lambda x : x // 11)

    skf = StratifiedKFold(n_splits=train_config["num_folds"])
    for fold, (_, val_index) in enumerate(skf.split(train_df, train_df.bucket)):
        train_df.loc[val_index, "fold"] = fold

    
    return train_df, test_df


def get_dataloaders(train_df, fold, train_config):
    train_df["is_valid"] = (train_df["fold"] == fold).values
    return ImageDataLoaders.from_df(train_df,
                           fn_col="path", label_col="score",
                           y_block=RegressionBlock, valid_col="is_valid", num_workers=train_config["num_workers"],
                           item_tfms=Resize(384), batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation(), Normalize.from_stats(*imagenet_stats)]), 
                            bs=4, val_bs=4, device=torch.device("cuda") )