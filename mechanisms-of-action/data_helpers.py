import pandas as pd
import numpy as np
from path import Path

from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import QuantileTransformer

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def add_fold(train_features, train_targets_scored, num_folds=5):
    mskf = MultilabelStratifiedKFold(n_splits=num_folds)
    
    for fold, (train_indices, val_indices) in enumerate(mskf.split(train_features, y=train_targets_scored)):
        train_features.loc[val_indices, "fold"] = fold
        train_targets_scored.loc[val_indices, "fold"] = fold
    
    train_features.iloc[:,4:] = train_features.iloc[:,4:].astype(np.float32)
    train_targets_scored.iloc[:,1:] = train_targets_scored.iloc[:,1:].astype(np.float32)
    train_features = train_features.drop("cp_type", axis=1)
    
    return train_features, train_targets_scored

def add_PCA(df, columns, n_components, name):
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(df[columns].values)
    extra_df = pd.DataFrame(data=X, columns=[name + str(i) for i in range(n_components)])
    
    df = pd.concat([df, extra_df], axis=1)    
    return df   

def one_hot(df):
    df["cp_time"] = df["cp_time"].astype(str)
    df = df.set_index("sig_id")
    
    df = pd.get_dummies(df)
    df = df.reset_index()
    return df

def normalize_columns(df):
    original_genes = [c for c in df.columns if c.startswith("g-")]
    original_cells = [c for c in df.columns if c.startswith("c-")]
    
    new_genes = [c for c in df.columns if c.startswith("G-")]
    new_cells = [c for c in df.columns if c.startswith("C-")]
    
    columns = [*original_genes, *original_cells, *new_genes, *new_cells]
    
    quantile_transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
    df[columns] = quantile_transformer.fit_transform(df[columns].values)
    
    return df

def remove_id_and_fold(df):
    df = df.drop(["sig_id", "fold"], axis=1, errors="ignore")
    
    return df
    
def remove_low_variance_features(df, threshold=0.8):    
    variance_threshold = VarianceThreshold(threshold=threshold)
    
    base_df  = df[["sig_id", "cp_time", "cp_dose", "fold"]]
    original_genes_df = df[ [c for c in df.columns if c.startswith("g-")] ]
    original_cells_df = df[[c for c in df.columns if c.startswith("c-")]]
    
    genes_array = variance_threshold.fit_transform(df[[c for c in df.columns if c.startswith("G-")]])
    cells_array = variance_threshold.fit_transform(df[[c for c in df.columns if c.startswith("C-")]])
    
    genes_pca_df = pd.DataFrame(data=genes_array, columns= ["G-" + str(i) for i in range(genes_array.shape[1])] )
    cells_pca_df = pd.DataFrame(data=cells_array, columns= ["C-" + str(i) for i in range(cells_array.shape[1])] )
    
    df = pd.concat([base_df, original_genes_df, original_cells_df, genes_pca_df, cells_pca_df], axis=1)

    return df

def eliminate_rows_based_on_cp_type(train_df, train_targets_scored):
    reduction_array = (train_df.cp_type != "ctl_vehicle")
    train_df_reduced = train_df[reduction_array].reset_index(drop=True)
    train_targets_scored_reduced = train_targets_scored[reduction_array].reset_index(drop=True)
    
    return train_df_reduced, train_targets_scored_reduced

def get_cp_type_col():
    test_features = pd.read_csv(Path(config["path_to_data_dir"])  / "test_features.csv")
    return test_features["cp_type"]

def train_val_split(features, targets, val_fold=4):
    train_features = features[features.fold != val_fold]
    train_targets = targets[targets.fold != val_fold]
    
    val_features = features[features.fold == val_fold]
    val_targets = targets[targets.fold == val_fold]    
    
    return train_features, val_features, train_targets, val_targets