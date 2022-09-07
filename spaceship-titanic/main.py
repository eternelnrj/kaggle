from data_functions import create_df
from training import Fitter, cross_validation


train_config = {"eta" : 0.04, "n_estimators" : 1000, "early_stopping_rounds" : 50, "n_jobs" : -1,
 "max_depth" : 4}

df = create_df()

processed_train_df = df[df.Where == "train"].drop(["Where", "PassengerId"], axis=1)
processed_test_df = df[df.Where == "test"].drop(["Where", "Transported", "fold"], axis=1)

cross_validation(processed_train_df, processed_test_df.drop('PassengerId',axis=1), train_config)
fitter = Fitter(processed_train_df, processed_test_df.drop('PassengerId', axis=1), train_config)
fitter.train()

submission = fitter.get_submission(processed_test_df)
submission.to_csv("submission.csv", index=False)