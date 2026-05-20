"""
Processes embedding dictionaries into dataframes for lexical gap experiment pipeline.

The processing instructions will vary according to how embedding vectors are stored post retrieval.
For example of how embedding dictionaries were stored for this process, please see folder (pending).

The output of this process is two files:
- DF_MAP: contains source-target word and vector mappings, along with model family and retrieval
          type attributes, along with identifiers for downstream filtering for 1:1 and many:1 only.
- DF_GAP: contains the same structure, with source words only (no known translations).

Please see folder (pending) for example DF_MAP and DF_GAP files used for the experiment pipeline.
"""

from utils import save, setup_logging
from config import CONFIG
import pandas as pd
import torch
import logging
logger = logging.getLogger(__name__)
setup_logging(CONFIG['paths']['log_dir'])


def load_embedding_dict(file_name):
    ''' 
    Load PyTorch dictionary from local filepath

    Args:
        file_name: local file name

    Returns:
        Nested embedding dictionary
    '''
    return torch.load(
        CONFIG['paths']['embedding_dicts'] + '/' + file_name, map_location=torch.device('cpu'))


def prepare_vec(x: torch.Tensor) -> torch.Tensor:
    '''
    Normalize embedding tensor into 1D vector
    Casts to float32
    Mean pools across token dimension if 3D
    Squeezes leading batch dim of 1 if present

    Args:
        x: embedding tensor of shape (1, tokens, d), (1, d), or (d,)

    Returns:
        1D tensor of shape (d,)

    Raises:
        ValueError: if tensor can not be reduced to 1D
    '''
    x = x.to(torch.float32)
    if x.dim() == 3:
        # assume (batch, tokens, d) or (1,tokens,d)
        x = x.mean(dim=1)
    if x.dim() == 2 and x.shape[0] == 1:
        x = x[0]
    if x.dim() != 1:
        raise ValueError(f"Expected 1D vector after preparation, got shape {tuple(x.shape)}")
    return x


def flatten_src_tgt_dict(model_dict, model_name, vector_type, src_language):
    '''
    For the src-tgt dict mapping, convert each src-tgt word map to
        [model name, vector type, src language, src word, tgt word, src vec, tgt vec]

    Args:
        model_dict: nested embedding dictionary
        model_name: alias for final dataframe
        vector_type: first/last/last4/sentence, etc
        src_language: (korean, english)

    Returns:
        Nested list with each entry a source-target word pair, along with corresponding embeddings.
    '''

    src_words = [k for k, v in model_dict.items() if v.get('language').lower() == src_language]
    src_lang, tgt_lang = {'korean': ('kor', 'eng'), 'english': ('eng', 'kor')}[src_language]
    results = []
    for src_word in src_words:
        src_vec = prepare_vec(model_dict[src_word][src_lang + vector_type])
        for tgt_word, info in model_dict[src_word][tgt_lang + '_words'].items():
            tgt_vec = prepare_vec(info[tgt_lang + vector_type])
            results.append(
                [model_name, vector_type, src_language, src_word, tgt_word, src_vec, tgt_vec])

    return results


def flatten_src_gapword_dict(model_dict, model_name, vector_type, src_language):
    '''
    For the gap dict mapping, convert each src word to
        [model name, vector type, src language, src word, src vec]

    Args:
        model_dict: nested embedding dictionary
        model_name: alias for final dataframe
        vector_type: first/last/last4/sentence, etc
        src_language: (korean, english)

    Returns:
        Nested list with each entry a source word and corresponding embedding.
    '''

    src_words = [k for k, v in model_dict.items() if v.get('language').lower() == src_language]
    results = []
    for src_word in src_words:
        src_vec = prepare_vec(model_dict[src_word][vector_type])
        results.append([model_name, '_'+vector_type, src_language, src_word, src_vec])

    return results

# Create DF_MAP:
# Dataframe of all source words and embedding vector, and their target translation word and vector.

# load nested dictionaries
d_exa_24_ia = load_embedding_dict('iam_embed_adj_paired_exaone_3.5_2.4b_updated.pt')
d_exa_78_ia = load_embedding_dict('iam_embed_adj_paired_exaone_3.5_7.8b_updated.pt')
d_kn_na_ia = load_embedding_dict('iam_embed_adj_paired_kanana_nano_embed_updated.pt')
d_kn_21_ia = load_embedding_dict('iam_embed_adj_paired_kanana_1.5_2.1b_base_updated.pt')
d_kn_8_ia = load_embedding_dict('iam_embed_adj_paired_kanana_1.5_8b_base_updated.pt')

# set configuration for models and vector types
configs = [
    (d_exa_24_ia,   'd_exa_24_ia',   ['_vector', '_vector_last4', '_vector_first']),
    (d_exa_78_ia,   'd_exa_78_ia',   ['_vector', '_vector_last4', '_vector_first']),
    (d_kn_na_ia,    'd_kn_na_ia',    ['_vector', '_vector_last4', '_vector_first', '_sent_vector']),
    (d_kn_21_ia,    'd_kn_21_ia',    ['_vector', '_vector_last4', '_vector_first']),
    (d_kn_8_ia,     'd_kn_8_ia',     ['_vector', '_vector_last4', '_vector_first'])
    ]

# flatten all configs
final = []
for model_dict, model_name, vector_type in configs:
    for vt in vector_type:
        for src_language in ['korean', 'english']:
            final.extend(
                flatten_src_tgt_dict(
                    model_dict=model_dict,
                    model_name=model_name,
                    vector_type=vt,
                    src_language=src_language)
            )

group_cols = ['model_version', 'vector_type', 'language', 'src_word']
other_cols = ['tgt_word', 'src_vec', 'tgt_vec']
DF_MAP = pd.DataFrame(final, columns=group_cols+other_cols)

# key identifier to ensure 1:1 and many:1 only downstream
DF_MAP['n_tgt_words'] = DF_MAP.groupby(group_cols)['tgt_word'].transform('count')
DF_MAP['is_only_tgt_word'] = (DF_MAP['n_tgt_words'] == 1).astype(bool)

# remove any rows where src or tgt vector has nulls
bad_mask = (
    DF_MAP.src_vec.apply(lambda x: (~torch.isfinite(x)).any().item()) |
    DF_MAP.tgt_vec.apply(lambda x: (~torch.isfinite(x)).any().item())
)
DF_MAP = DF_MAP.loc[~bad_mask].reset_index(drop=True)


# Create DF_GAP:
# Dataframe of all source words and embedding vector with no known translations.
# This is used for AUC analysis and model training. 

# load nested dictionaries
d_exa_24_ia = load_embedding_dict('iam_embed_adj(gap)_exaone_3.5_2.4b_updated.pt')
d_exa_78_ia = load_embedding_dict('iam_embed_adj(gap)_exaone_3.5_7.8b_updated.pt')
d_kn_na_ia = load_embedding_dict('iam_embed_adj(gap)_kanana_nano_embed_updated.pt')
d_kn_21_ia = load_embedding_dict('iam_embed_adj(gap)_kanana_1.5_2.1b_base_updated.pt')
d_kn_8_ia = load_embedding_dict('iam_embed_adj(gap)_kanana_1.5_8b_base_updated.pt')

# set configuration for models and vector types
configs = [
    (d_exa_24_ia,      'd_exa_24_ia',      ['embed', 'embed_last4', 'embed_first']),
    (d_exa_78_ia,      'd_exa_78_ia',      ['embed', 'embed_last4', 'embed_first']),
    (d_kn_na_ia,       'd_kn_na_ia',       ['embed', 'embed_last4', 'embed_first', 'sent_embed']),
    (d_kn_21_ia,       'd_kn_21_ia',       ['embed', 'embed_last4', 'embed_first']),
    (d_kn_8_ia,        'd_kn_8_ia',        ['embed', 'embed_last4', 'embed_first'])
    ]

# flatten all configs across Korean and English
final = []
for model_dict, model_name, vector_type in configs:
    for vt in vector_type:
        for src_language in ['korean', 'english']:
            final.extend(
                flatten_src_gapword_dict(
                    model_dict=model_dict,
                    model_name=model_name,
                    vector_type=vt,
                    src_language=src_language)
            )

# create dataframe, align vector_type naming with DF_MAP ("vector" instead of "embed")
DF_GAP = pd.DataFrame(
    final,
    columns=['model_version', 'vector_type', 'language', 'src_word', 'src_vec'])
DF_GAP['vector_type'] = DF_GAP.vector_type.apply(lambda x: x.replace('embed', 'vector'))

# removes non-finite (NaN)
bad_gap = DF_GAP.src_vec.apply(lambda x: (~torch.isfinite(x)).any().item())

DF_GAP = DF_GAP.loc[~bad_gap].reset_index(drop=True)

# Save

# check to remove all gap words from map df
DF_MAP = DF_MAP[~DF_MAP.src_word.isin(list(set(DF_GAP.src_word)))]

# convert to numpy
DF_MAP['src_vec'] = DF_MAP['src_vec'].apply(lambda v: v.cpu().numpy())
DF_MAP['tgt_vec'] = DF_MAP['tgt_vec'].apply(lambda v: v.cpu().numpy())
DF_GAP['src_vec'] = DF_GAP['src_vec'].apply(lambda v: v.cpu().numpy())

# save to local filepath
save(DF_MAP, CONFIG['paths']['df_save'] + '/cg_df_map.pkl')
save(DF_GAP, CONFIG['paths']['df_save'] + '/cg_df_gap.pkl')
