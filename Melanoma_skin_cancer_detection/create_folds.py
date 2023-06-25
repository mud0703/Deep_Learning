import os
import pandas as pd 
from sklearn import model_selection

if __name__ == '__main__':
    input_path = 'Melanoma_skin_cancer_detection/data/melanoma-classification/'
    df = pd.read_csv(os.path.join(input_path, 'train.csv'))
    df['k_fold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf =model_selection.StratifiedKFold(n_splits=10)
    for fold, (_, _) in enumerate(kf.split(X=df, y=y)):
        df.loc[:, 'k_fold'] = fold
    df.to_csv(os.path.join(input_path, 'train_fold.csv'), index=False)

