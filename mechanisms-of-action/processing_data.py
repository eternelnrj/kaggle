import pandas as pd
import numpy as np
from path import Path

from config import config
from data_helpers import add_fold, add_PCA, remove_low_variance_features, eliminate_rows_based_on_cp_type, one_hot, normalize_columns


def create_df_and_targets(train_config):
    train_df = pd.read_csv(Path(config["path_to_data_dir"])  / "train_features.csv")
    train_targets_scored = pd.read_csv(Path(config["path_to_data_dir"]) / "train_targets_scored.csv")
    
    train_df, train_targets_scored = eliminate_rows_based_on_cp_type(train_df, train_targets_scored)
    train_df, train_targets_scored = add_fold(train_df, train_targets_scored, train_config.num_folds)
    
    test_df = create_test_df()
    full_df = concatenate_and_process(train_df, test_df)
    
    return full_df, train_targets_scored

def create_test_df():
    test_features = pd.read_csv(Path(config["path_to_data_dir"]) / "test_features.csv")
    
    test_features = test_features.drop("cp_type", axis=1)    
    test_features.iloc[:,3:] = test_features.iloc[:,3:].astype(np.float32)
    
    return test_features


def concatenate_and_process(train_df, test_df):
    train_size = train_df.shape[0]

    df = pd.concat([train_df, test_df], axis=0)
    df = df.reset_index(drop=True)
    
    genes_cols = [c for c in df.columns if c.startswith("g-")]
    cells_cols = [c for c in df.columns if c.startswith("c-")]
    
    df = add_PCA(df, genes_cols, n_components=600, name="G-")
    df = add_PCA(df, cells_cols, n_components=50, name="C-")
    df = remove_low_variance_features(df, threshold=0.8)
    df = one_hot(df)
    
    df = df.reset_index(drop=True)
    df.loc[:train_size-1, "WHERE"] = "train"
    df.loc[train_size:, "WHERE"] = "test"
    
    fold = df.fold
    df = df.drop("fold", axis=1)
    df["fold"] = fold
    
    df = normalize_columns(df) 
        
    return df
