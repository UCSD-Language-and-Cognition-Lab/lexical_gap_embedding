'''
List of configurations for experiment runs, along with filepaths for input/output.

Each experiment run is for one source language at a time. 

LANGUAGE: source language for experiment. 
pct_train: Percent of source words to use in training set
pca_dim: Number of dimensions produced via PCA for dimensionality reduction
csls_k: Number of neighbors to use for CSLS calculation
num_iters: Number of random seeds to run
'''

# Set separately per language (korean, english), one full pipeline run per language. 
LANGUAGE = 'korean'

LANGUAGE_PATHS = {'korean': 'K', 'english': 'E'}
LANGUAGE_PATH = LANGUAGE_PATHS[LANGUAGE]

CONFIG = {
    # file paths for input / output
    'paths': {
        'embedding_dicts': 'data/input/embedding_dicts',
        'df_save': 'data/input',
        'df_map': 'data/input/cg_df_map.pkl',
        'df_gap': 'data/input/cg_df_gap.pkl',
        'pipeline_output': f'data/output/{LANGUAGE_PATH}',
        'log_dir': '_logs',
    },
    # run parameters (percent train/test split, PCA dimensions, csls neighbors, number of seeds).
    'run': {
        'src_language': LANGUAGE,
        'pct_train': 0.8,
        'pca_dim': 256,
        'csls_k': 10,
        'num_iters': 100,
    },
    # naming and boolean flag for transformation / dimensionality reduction.
    'tf_dm_sets': [
        {'name': 'unaligned-alldims', 'procrustes_flag': False, 'pca_flag': False},
        {'name': 'unaligned-pca', 'procrustes_flag': False, 'pca_flag': True},
        {'name': 'procrustes-alldims', 'procrustes_flag': True, 'pca_flag': False},
        {'name': 'procrustes-pca', 'procrustes_flag': True, 'pca_flag': True},
    ],
    # list of model families and vector types (from DF_MAP and DF_GAP input DataFrames).
    'embedding_multiverse': [
        {'model_family': 'd_kn_na', 'vector_type': '_sent_vector'},
        {'model_family': 'd_kn_na', 'vector_type': '_vector'},
        {'model_family': 'd_kn_na', 'vector_type': '_vector_last4'},
        {'model_family': 'd_kn_21', 'vector_type': '_vector_first'},
        {'model_family': 'd_kn_8', 'vector_type': '_vector_first'},
        {'model_family': 'd_exa_78', 'vector_type': '_vector_last4'},
        {'model_family': 'd_exa_78', 'vector_type': '_vector_first'},
        {'model_family': 'd_exa_24', 'vector_type': '_vector'},
        {'model_family': 'd_exa_24', 'vector_type': '_vector_last4'},
        {'model_family': 'd_exa_24', 'vector_type': '_vector_first'},
    ],
}