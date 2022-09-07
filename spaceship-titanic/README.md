## Spaceship Titanic Competition
In this solution use `xgboost classifier` combined with `pseudo labelling`. Note that during cross validation the validation data is not used to compute pseudo labels, which reduces the risk of a common problem with pseudo labeling: over optimistic cross validation score. A solid `0.802+ accuracy score` on the whole data follows from cross validation with a rather simple model, since we do not use the text data provided.

## Data
Data can be downloaded from `https://www.kaggle.com/competitions/spaceship-titanic/data`.

## Usage
`python3 main.py` to train the model and save predictions in a .csv file.




