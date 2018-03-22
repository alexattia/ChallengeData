import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import gc

all_features = ['assist', 'bad pass', 'block', 'defensive foul',
       'defensive rebound', 'fg', 'lost ball', 'miss', 'offensive foul',
       'offensive rebound', 'score', 'steals', 'three pts',
       'total rebound', 'two pts']

def add_incremental_features(df):
    """
    Compute total rebounds, turnovers, FGA stats over the game
    :param df: dataframe
    :return: dataframe with new column
    """
    for k in tqdm(range(1,1441)):
        df['total rebound_%d' % k] = df['offensive rebound_%d' % k] + df['defensive rebound_%d' % k]
        df['turnover_%d' % k] = df['bad pass_%d' % k] + df['lost ball_%d' % k] + df['offensive foul_%d' % k]
        df['fga_%d' % k] = df['fg_%d' % k] + df['miss_%d' % k] + df['block_%d' % k]
    return df

def convert_pt(x, nb):
    """
    Convert score difference into 2/3 pts goals
    :param x: value, pandas columns
    :param nb: 2 or 3
    :return: differential value
    """
    if np.abs(x) == nb:
        return np.sign(x)
    else:
        return 0

def add_fg(df, test=False):
    """
    Compute 2pts, 3pts and FGs stats over the game
    :param df: dataframe
    :param test: boolean for test dataframe to not consider label
    :return: dataframe with new columns
    """
    # Compute score difference
    df['diff score_%d' % 1] = df['score_%d' % 1]
    for k in tqdm(range(2,1441)):
        df['diff score_%d' % k] = df['score_%d' % k] - df['score_%d' % (k-1)]
    # Compute 2pts/3pts
    for k in tqdm(range(1,1441)):
        df['two pts_%d' % k] = df['diff score_%d' % k].apply(lambda x:convert_pt(x, 2))
        df['three pts_%d' % k] = df['diff score_%d' % k].apply(lambda x:convert_pt(x, 3))
    # Compute 2pts/3pts as cumulative sum
    two_pts_df = df[[k for k in df.columns if 'two pts' in k]].cumsum(axis=1)
    three_pts_df = df[[k for k in df.columns if 'three pts' in k]].cumsum(axis=1)
    df = pd.concat([df[[k for k in df.columns if 'two pts' not in k and 'three pts' not in k]], 
                        two_pts_df,
                        three_pts_df], axis=1)
    # Compute FG
    for k in tqdm(range(1,1441)):
        df['fg_%d' % k] = df['two pts_%d' % k] + df['three pts_%d' % k]
    
    if test:
        keep = ['ID']
    else:
        keep = ['ID', 'label']
    return df[keep + sorted([k for k in df.columns if len(k.split('_')) > 1 and 'diff ' not in k], 
                                       key=lambda x:int(x.split('_')[1]))]

def compute_absolute(df):
    ki = ['score', 'assist', 'fg', 'fga', 'three pts', 'turnover']
    scores = df[[k for k in df.columns if any(p in k for p in ki) and 'up' not in k]]
    for var in ki:
        temp_away, temp_home = np.zeros((len(scores), 1440)), np.zeros((len(scores), 1440))
        for k in range(1,1441):
            if k > 1:
                diff = (scores['%s_%d' % (var, k)] - scores['%s_%d' % (var, k-1)]).values
            else:
                diff = scores['%s_%d' % (var, k)].values
            temp_away[np.flatnonzero(diff>0), k-1] = diff[np.flatnonzero(diff>0)]
            temp_home[np.flatnonzero(diff<0), k-1] = diff[np.flatnonzero(diff<0)]
        df['%s_away_final' % var] = temp_away.sum(axis=1)
        df['%s_home_final' % var] = np.abs(temp_home.sum(axis=1))
    temp = len(df)
    # filter aberrant data
    df = df[df['score_away_final'] >= 19]
    df = df[df['score_home_final'] >= 19]
    print('%d samples have been removed - weird number of points' % (temp-len(df)))
    return df

def compute_advanced_stats(df):
    teams = ['home', 'away']
    for team in teams:
        df['assist_turnover_%s' % team] = df['assist_%s_final' % team] / df['turnover_%s_final' % team]
        df['assist_ratio_%s' % team] = df['assist_%s_final' % team] / (df['fga_%s_final' % team] + df['assist_%s_final' % team] + df['turnover_%s_final' % team])
        df['fg_percentage_%s' % team] = df['fg_%s_final' % team] / df['fga_%s_final' % team]
        df['effective_fg_%s' % team] = (df['fg_%s_final' % team] + 0.5*df['three pts_%s_final' % team]) / df['fga_%s_final' % team]
    return df

def add_features_24(df):
    for var in tqdm(all_features):
        temp = df[[k for k in df.columns if var in k and int(k.split('_')[1]) % 24 == 0]].diff(axis=1)
        temp.columns = [k + '_up' for k in temp.columns]
        temp = temp[temp.columns[1:]]
        df = pd.concat([df, temp], axis=1)
        del temp
        gc.collect()
    return df