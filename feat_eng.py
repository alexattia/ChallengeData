import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm

def add_tot_rebounds(df):
    """
    Compute total rebounds stats over the game
    :param df: dataframe
    :return: dataframe with new column
    """
    for k in range(1,1441):
        df['total rebound_%d' % k] = df['offensive rebound_%d' % k] + df['defensive rebound_%d' % k]
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