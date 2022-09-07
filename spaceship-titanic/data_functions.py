import pandas as pd
from path import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from config import config

def create_df():
    data_folder = Path(config["path_to_data_dir"])
    train_path = data_folder / "train.csv"
    test_path = data_folder / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    df = process_df(train_df, test_df)

    return df
######################################################################################################
def process_df(train, test):
    train = process_labels(train)
    train = add_folds(train)
    
    df = merge_train_test(train, test)
    
    df = process_object_columns(df)
    df = process_numerical_columns(df)
    df = process_boolean_columns(df)
    
    df = drop_useless_columns(df)
    
    return df

def process_labels(df):
    df["Transported"] = df["Transported"].apply(lambda x : int(x))
    return df

def add_folds(df):
    df["Group"] = df.PassengerId.apply(lambda x : int(x.split("_")[1]) )
    sgkf = StratifiedGroupKFold(n_splits=5)
    
    for fold, (_, val_idxs) in enumerate(sgkf.split(df, y=df.Transported, groups=df["Group"].values)):
        df.loc[val_idxs, "fold"] = fold
    
    return df

def merge_train_test(train, test):
    train["Where"] = "train"
    test["Where"] = "test"
    
    df = pd.concat([train, test], axis=0)
    
    return df
######################################################################################################
def process_object_columns(df):
    columns = ["HomePlanet","Destination"]
    le = LabelEncoder()
    
    for column in columns:
        df[column] = df[column].fillna("Unknown")
        df[column] = le.fit_transform(df[column])

    df["Cabin"] = df["Cabin"].fillna("Unknown/Unknown/Unknown")
    
    df = add_side(df)
    df = add_side_2(df)
    
    return df

def process_numerical_columns(df):
    columns = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    for column in columns:
        df[column] = df[column].fillna(df[column].mean())
    
    df["TotalCost"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"] 

    return df
    
def process_boolean_columns(df):
    columns = ["CryoSleep", "VIP"]
    
    le = LabelEncoder()
    
    for column in columns:
        df[column] = df[column].fillna("Unknown")
        df[column] = df[column].apply(lambda x : str(x))
        df[column] = le.fit_transform(df[column])
    
    return df
    
def drop_useless_columns(df):
    df = df.drop("Name", axis=1)
    df = df.drop("Group", axis=1)
    df = df.drop("Cabin", axis=1)
    
    return df

def add_side(df):
    df["Side"] = df["Cabin"].apply(lambda x : x.split("/")[-1])
    
    le = LabelEncoder()
    df["Side"] = le.fit_transform(df["Side"])

    return df


def add_side_2(df):
    df["Side_2"] = df["Cabin"].apply(lambda x : x.split("/")[0])
    
    le = LabelEncoder()
    df["Side_2"] = le.fit_transform(df["Side_2"])

    return df
######################################################################################################
def train_val_split_by_fold(df, fold):
    val_df = df[df["fold"] == fold]
    train_df = df[df["fold"] != fold]
    
    train_df = train_df.drop("Where", axis=1, errors='ignore')
    val_df = val_df.drop("Where", axis=1, errors='ignore')
            
    return train_df, val_df