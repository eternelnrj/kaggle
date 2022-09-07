## PetFinder.my - Pawpularity Contest
See `https://www.kaggle.com/competitions/petfinder-pawpularity-score` for details about this competition.  For each photo of a pet we need to predict its pawpularity - a number from 0 to 100, which arises from pet's profile view statistics and many other metrics we don't know. Submissions are scored on the root mean squared error. This is a reconstruction of my solution, which gives a score of approximately `17.04`, which corresponds to the silver medal area. This result is mainly possible thanks to the use of an efficient transformer (transformers turn out to perform better than CNN in this contest) and test time augmentations. 

## Data
Data can be downloaded from `https://www.kaggle.com/competitions/petfinder-pawpularity-score/data`.

## Usage
Create `config.py` with `config = {"path_to_data_dir" : "PATH_TO_YOUR_DATA_DIR"}` inside. After installing the required packages use the command `python3 main.py` to train the model and save predictions in a .csv file.




