from sklearn.metrics import accuracy_score
from data_functions import train_val_split_by_fold
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from tqdm import tqdm


def cross_validation(df, processed_test_df, config):
    full_predictions = []
    full_targets = []
    
    for fold in df.fold.unique():
        train_df, val_df = train_val_split_by_fold(df, fold)

        fitter = Fitter(train_df, processed_test_df, config)
        fitter.train()

        predictions = fitter.predict(val_df.drop(["Transported", "fold"], axis=1))
        targets = val_df.Transported.values

        full_predictions.append(predictions)
        full_targets.append(targets)

        print(f"accuracy_score for fold {int(fold)}:", accuracy_score(targets, predictions))

    full_predictions = np.concatenate(full_predictions)
    full_targets = np.concatenate(full_targets)
    print()
    print("accuracy_score on the whole data: ", accuracy_score(full_targets, full_predictions))
    
    
class Fitter:
    def __init__(self, train_df, test_df, config):
                
        self.train_df = train_df
        self.test_df = test_df.copy()
        self.config = config
        self.classifiers = []

    def train(self, pseudolabels_df=None):
        self.classifiers = []
        
        for fold in tqdm(self.train_df.fold.unique()):
            clf = XGBClassifier(**self.config)
            train_df, val_df = train_val_split_by_fold(self.train_df, fold)
 
            if pseudolabels_df is not None:
                train_df = pd.concat([train_df, pseudolabels_df], axis=0)
            
            X_train = train_df.drop(["Transported", "fold"], axis=1)
            y_train = train_df.Transported
            
            X_val = val_df.drop(["Transported", "fold"], axis=1)
            y_val = val_df.Transported
            
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)     
                
            self.classifiers.append(clf)      
        
        if pseudolabels_df is None:
            self.test_df["Transported"] = self.predict_proba(self.test_df)
            self.test_df = self.test_df[(self.test_df["Transported"] > 0.97) ]
            self.test_df["Transported"] = np.rint(self.test_df["Transported"].values)
            self.train(self.test_df)
        
    def predict(self, test_df):
        predictions = []
    
        for classifier in self.classifiers:
            predictions.append(np.expand_dims(classifier.predict_proba(test_df)[:,1], axis=0))

        return (np.rint(np.concatenate(predictions, axis=0).mean(axis=0)) == 1)
    
    def predict_proba(self, test_df):
        predictions = []
    
        for classifier in self.classifiers:
            predictions.append(np.expand_dims(classifier.predict_proba(test_df)[:,1], axis=0))

        return (np.concatenate(predictions, axis=0).mean(axis=0))
    
    def get_submission(self, test_df):
        predictions = self.predict(test_df.drop("PassengerId", axis=1))
        submission = test_df[["PassengerId"]]
        submission["Transported"] = predictions
        
        return submission