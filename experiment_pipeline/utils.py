"""
File and logging utils for conceptual gap experiment pipeline.
"""

import pickle
from pathlib import Path
import logging
import numpy as np

def load(path: str):
    '''
    Load pickle file from local filepath.

    Args:
        path: local filepath

    Returns:
        loaded object
    '''    

    with open(path, 'rb') as f:
        return pickle.load(f)


def save(obj, path: str):
    '''
    Save python object to pickle file.

    Args:
        obj: python object to save
        path: local filepath
    '''    

    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def setup_logging(path: str, name='pipeline'):
    '''
    Setup local logging file.

    Args:
        path: local filepath
        name: name of pipeline file log
    '''     

    Path(path).mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(f'{path}/{name}.log', mode='w'),
            logging.StreamHandler(),
        ],
        force=True,
    )


def lists_to_vecs(df, vec_cols):
    '''
    Converts .parquet formatted embedding vectors to numpy arrays, for df_map and df_gap

    Args:
        df: DataFrame (df_map, df_gap)
        vec_cols: Columns in df with vectors (src_vec/tgt_vec for df_map, src_vec for df_gap)

    Returns:
        DataFrame with vector columns cast to np.float32 numpy arrays
    '''

    df = df.copy()

    for col in vec_cols:
        df[col] = df[col].apply(lambda v: np.asarray(v, dtype=np.float32))

    return df    