import feat_eng
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from time import time
import numpy as np
from xgboost import XGBClassifier

def load_dataset():
    train = pd.read_csv('./train.csv')
    label = pd.read_csv('./challenge_output_data_training_file_nba_challenge.csv', sep=';')
    train = pd.merge(train, label, on='ID')

    train = feat_eng.add_tot_rebounds(train)
    train = feat_eng.add_fg(train)
    print('Dataset loaded with new features')
    return train_test_split(train, test_size=0.2, random_state=42)

def compute_score(X, y, X_val, y_val, nb_iter):
    scores = []
    for k in range(nb_iter):
        print('Iteration %d/%d' % (k+1, nb_iter))
        start = time()
        params = {'base_score': 0.5,
                'booster': 'gbtree', 
                'colsample_bylevel':np.random.choice([0.4,0.7,1]),
                'colsample_bytree':np.random.choice([0.8,1]),
                'learning_rate':np.random.choice([0.01,0.05,0.1]),
                'min_child_weight':np.random.choice([1,2,5,10]),
                'max_delta_step':np.random.choice([0,1,2]),
                'max_depth': np.random.choice([2, 5, 8, 10]),
                'n_estimators':np.random.choice([100, 500, 1000])}
        xgb = XGBClassifier(**params)
        xgb.fit(X, y)
        y_pred = xgb.predict(X_val)
        scores.append([np.mean(y_pred == y_val), params, time()-start])
        print(np.mean(y_pred == y_val), params)
    return scores

def write_best(scores):
    scores = sorted(scores, key=lambda x:x[0], reverse=True)
    file = open("./grid_search.txt", 'w')
    for i, score in enumerate(scores):
        file.write('Model #%d computed in %d sec :' % (i+1, score[2]))
        file.write('OOB score : %.3f with params %s' % (score[0], score[1]))
        file.write('\n')
    file.close()

if __name__ == '__main__':
    # Open arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('nb_iter', help='number of iterations for grid_search')
    args = parser.parse_args()
    df_train, df_val = load_dataset()
    y = df_train.label.values
    X = df_train.drop(['ID', 'label'], axis=1)
    y_val = df_val.label.values
    X_val = df_val.drop(['ID', 'label'], axis=1)    
    scores = compute_score(X, y, X_val, y_val, int(args.nb_iter))
    write_best(scores)
