## Mechanisms of Action (MoA) Prediction Competition
For details about this contest, see `https://www.kaggle.com/competitions/lish-moa`. This is a multilabel classification problem and submissions are evaluated with the log loss metric. In this solution, we use a shallow neural network and label smoothing to avoid overfitting. This solution gives a score of around `0.01630`, which is relatively far from the medal area (< `0.01616`), but uses many essential ideas to avoid overfitting.

## Data
Data can be downloaded from `https://www.kaggle.com/competitions/lish-moa/data`.

## Usage
After installing the required packages use `python3 main.py` to train the model and save predictions in a .csv file.


