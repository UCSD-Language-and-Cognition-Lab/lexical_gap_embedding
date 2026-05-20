"""
File and logging utils for conceptual gap experiment pipeline.
"""

import pickle
from pathlib import Path
import logging

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