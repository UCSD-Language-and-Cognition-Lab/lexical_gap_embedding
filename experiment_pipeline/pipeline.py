"""
Experiment pipeline for cross lingual lexical gap identification.

Imports source-target word mapping from df_map and source words from df_gap, along with embedding
vectors, model family, embedding extraction method, and attributes for downstream filtering. 

Runs train-test splits across random seeds, produces multiple embedding spaces via dimensionality 
reduction and Procrustes alignment. Computes nearest-neighbor metrics across train, test, and
gap words at the word/trial level, and gap vs test AUC scores for each unique trial. Each pipeline
run is for one source language only. 

Results are saved to cg_df_trials and cg_df_auc for downstream analysis and modeling.
"""

from utils import load, save, setup_logging
from config import CONFIG
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import logging
logger = logging.getLogger(__name__)
setup_logging(CONFIG['paths']['log_dir'])


class TrainWarehouse:
    '''
    Computes PCA reduction, Procrustes transformation, mean centering and L2 norm for train set.
    Applies transformations to test and gap set.

    Args:
        src_vecs: Source language embeddings, shape (n_words, n_dims)
        tgt_vec: Target language embeddings, shape (n_words, n_dims)
        procrustes_flag: True for Procrustes orthogonal transformation, False for no transformation
        pca_flag: True for PCA dimensionality reduction, False for full dimensions
        pca_dim: Number of principal components for PCA reduction
    '''     

    def __init__(
        self,
        src_vecs: np.ndarray,
        tgt_vecs: np.ndarray,
        procrustes_flag: bool,
        pca_flag: bool,
        pca_dim: int | None
    ):

        self.procrustes_flag = procrustes_flag
        self.pca_flag = pca_flag
        self.pca_dim = pca_dim
        self.S = src_vecs.copy()
        self.T = tgt_vecs.copy()

    def fit(self) -> 'TrainWarehouse':
        '''
        Runs PCA and/or Procrustes based on class configuration. Saves source and target means for
        downstream train and gap vector projection.

        Unaligned-Alldims:    Mean center, L2 norm
        Unaligned-PCA:        PCA, Mean Center, L2 norm
        Procrustes-Alldims:   Mean center, L2 norm, Fit Procrustes
        Procrustes-PCA:       PCA, Mean center, L2 norm, Fit Procrustes

        Raises:
            AssertionError: If requested PCA dimensions > number of source+target samples. 
        '''

        if self.pca_flag:

            combined_data = np.vstack([self.S, self.T])
            
            n_samples, n_features = combined_data.shape
            assert self.pca_dim <= min(n_samples, n_features), \
                f"PCA dim {self.pca_dim} > min(n_samples={n_samples}, n_features={n_features})"
                
            self.pca = PCA(n_components=self.pca_dim, svd_solver='full')
            self.pca.fit(combined_data)
            
            # project Train S and T through new PCA space
            self.S = self.pca.transform(self.S)
            self.T = self.pca.transform(self.T)
 
        # save Train S and T means for downstream Train, Test, and Gap projection
        self.S_mean = self.S.mean(axis=0, keepdims=True)
        self.T_mean = self.T.mean(axis=0, keepdims=True)

        # mean center Train S and T
        self.S -= self.S_mean
        self.T -= self.T_mean
        # l2 norm Train S and T
        self.S /= np.linalg.norm(self.S, axis=1, keepdims=True) + 1e-12
        self.T /= np.linalg.norm(self.T, axis=1, keepdims=True) + 1e-12

        # compute procrustes W matrix based on Train S->T map
        if self.procrustes_flag:
            U, _, Vt = np.linalg.svd(self.S.T @ self.T)
            self.W = U @ Vt

        return self

    def project(self, X: np.ndarray, kind: str) -> np.ndarray:
        '''
        Projects vector through PCA and/or procrustes based on class configuration.
        Mean centers vector according to respective source/target mean.
        L2 norms vector.

        Args:
            X: vector for projection
            kind: 'source' or 'target'

        Returns:
            Projected vector

        Raises:
            AssertionError: Projection function call kind param not in ('source', 'target')
        '''           

        assert kind in ('source', 'target'), 'Kind must be source or target'
        
        # project through PCA (based on Train data)
        if self.pca_flag:
            
            X = self.pca.transform(X)

        # mean center (based on Train data)
        if kind == 'source':
            X = X - self.S_mean
        else:
            X = X - self.T_mean

        # l2 norm
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        # project through Procrustes (based on Train data)
        if self.procrustes_flag and kind == 'source':
            X = X @ self.W

        return X


def compute_csls(sim: np.ndarray, k: int) -> np.ndarray:
    '''
    Computes CSLS score matrix based on Cosine Similarity matrix. Corrects CS for hubness.
    2·cos(x,y) - cs_K(x) - cs_K(y)
    where cs_K() is the mean cosine similarity to the k-nearest neighbors in the other space

    Args:
        sim: matrix of pairwise cosine similarities
        k: number of neighbors for CSLS partition

    Returns:
        CSLS score matrix
    '''       

    # Calculate mean cs of source word to k nearest target words
    mean_cs_src_to_tgt = np.mean(np.partition(sim, -k, axis=1)[:, -k:], axis=1)

    # Calculate mean cs of target word to k nearest source words
    mean_cs_tgt_to_src = np.mean(np.partition(sim.T, -k, axis=1)[:, -k:], axis=1)

    # (2 * raw cosine similarity) - (1 * source hubness penalty) - (1 * target hubness penalty)
    return 2 * sim - mean_cs_src_to_tgt[:, None] - mean_cs_tgt_to_src[None, :]


def run_evaluations(
        X_words: np.ndarray,
        X_words_translation: np.ndarray,
        Y_words: np.ndarray,
        csls_sim: np.ndarray,
        cs_sim: np.ndarray,
        eval_type: np.ndarray
) -> list[dict]:
    '''
    For each word in the source set, compute the following metrics:
        - actual_translation_rank_csls: Rank of actual target based on highest CSLS score
        - neighbor_1_csls: CSLS score for closest neighbor
        - actual_translation_rank_cs: Rank of actual target based on highest CS score
        - neighbor_1_cs: CS score for closest neighbor
    CSLS metrics used for all analysis, neighbor_1_cs used for precision@1 comparison only.

    Args:
        X_words: train, test, and gap source words
        X_words_translation: train and test translations of source words + None for gap word.
        Y_words: train and test target words, deduplicated (search space)
        csls_sim: pairwise CSLS score matrix
        cs_sim: pairwise CS score matrix
        eval_type: train, test, or gap assignment ('train', 'test', 'gap')

    Returns:
        List of dictionaries for eval type, word, and associated metrics.
        To be concatenated into final dataframe across experiments.

    Raises:
        AssertionError: ensures no duplicate target words found for a source word (redundant check).
    '''     
    
    rows = []
    
    # for each source word
    for i, word in enumerate(X_words):      

        # Sort indices of target words by descending CSLS similarity
        neighborhood_idx_csls = np.argsort(-csls_sim[i])
        # First index (most similar)
        neighbor_1_idx_csls = neighborhood_idx_csls[0]
        # csls score for most similar
        neighbor_1_csls = csls_sim[i, neighbor_1_idx_csls]

        # Sort indices of target words by descending CS similarity
        neighborhood_idx_cs = np.argsort(-cs_sim[i])
        # First index (most similar)
        neighbor_1_idx_cs = neighborhood_idx_cs[0]
        # cs score for most similar
        neighbor_1_cs = cs_sim[i, neighbor_1_idx_cs]        

        # If has known translation
        t = X_words_translation[i]
        has_sot = isinstance(t, str) and len(t) > 0
        if has_sot:
            # Index of known translation in target universe 
            actual_translation_idxs = np.where(Y_words == X_words_translation[i])[0]
            # Redundant check on duplicates
            assert len(actual_translation_idxs) == 1, (
                f"Expected unique target word, got {len(actual_translation_idxs)} "
                f"for '{X_words_translation[i]}'"
            )
            actual_translation_idx = actual_translation_idxs[0]

            # Rank of known translation across all target word csls scores
            actual_translation_rank_csls = \
                int(np.where(neighborhood_idx_csls == actual_translation_idx)[0][0] + 1)

            # Rank of known translation across all target word cs scores
            actual_translation_rank_cs = \
                int(np.where(neighborhood_idx_cs == actual_translation_idx)[0][0] + 1)
          
        else:
            # Otherwise (if gap word), translation word metrics are None.
            actual_translation_rank_csls = actual_translation_rank_cs = None
        
        # Append dictionary to output list
        row = {
            'eval_type': eval_type[i],
            'src_word': word,

            'actual_translation_rank_csls': actual_translation_rank_csls,
            'neighbor_1_csls': neighbor_1_csls,
            'actual_translation_rank_cs': actual_translation_rank_cs,
            'neighbor_1_cs': neighbor_1_cs
            
        }
        rows.append(row)

    return rows


def auc_gap_vs_control(gap_metric_values: np.ndarray, test_metric_values: np.ndarray) -> float:
    '''
    Computes AUC (area under the curve) score for distribution of gap and test metrics, treating
    Test metrics as positive class (higher AUC = gap words have lower CSLS score than test words).

    Args:
        gap_metric_values: Metric values (CSLS or Cosine Similarity) of gap words
        test_metric_values: Metric values (CSLS or Cosine Similarity) of test_words

    Returns:
        AUC score for distribution
    '''       
    
    metric_values = np.concatenate([gap_metric_values, test_metric_values])

    # AUC is probability that gap word metric < test word metric 
    # (test words are 1, gap words are 0)
    labels = np.concatenate([
        np.zeros(len(gap_metric_values)),
        np.ones(len(test_metric_values))
    ])

    # Compute AUC score for test and gap word distribution
    return roc_auc_score(labels, metric_values)


def compute_aucs(df_trials: pd.DataFrame) -> pd.DataFrame: 
    '''
    Computes AUC score for each Gap/Test metric distribution, for each combination of parameters.

    Args:
        df_trials: DataFrame hosting all trial parameters, gap and test words, and metric values

    Returns:
        DataFrame of all experiment configurations and AUCs    
    '''        
    
    rows = []

    # For each unique trial
    for keys, df_trial in \
        df_trials.groupby(['model_family', 'vector_type', 'tf_dm', 'random_seed']):
        auc = auc_gap_vs_control(
            gap_metric_values = df_trial[df_trial.eval_type == 'gap']['neighbor_1_csls'].values,
            test_metric_values = df_trial[df_trial.eval_type == 'test']['neighbor_1_csls'].values) 
        # Append configuration and random seed number to output dictionary
        rows.append({
            **dict(zip(['model_family', 'vector_type', 'tf_dm', 'random_seed'], keys)),
            'metric': 'neighbor_1_csls',
            'auc': auc})
    df_auc = pd.DataFrame(rows)

    return df_auc


def main() -> None:
    '''
    Main experiment loop: iterate over all experiment configurations and trial seeds.

    The experiment explores a combination of configurations:
    1. Random seed train-test split, n=100
    2. Model family + vector type (d_kn_na sent_vector, ...), n=10
    3. Embedding Transformation / Dim reduction combinations (procrustes-alldims, ...), n=4
    4. CSLS based neighborhood metrics (similarity)

    Results are stored into single dataframe and saved.
    '''

    # Load SOT translation mapping (Train + Test) dataframe
    DF_MAP = load(CONFIG['paths']['df_map'])
    # Ensure each source word has 1 and only 1 target word (1:1, many:1)
    # (source words have only 1 target word, target word can map to multiple source words)
    DF_MAP = DF_MAP[DF_MAP.is_only_tgt_word].copy()

    # Load SOT gap word dataframe
    DF_GAP = load(CONFIG['paths']['df_gap'])

    # Run one source language per experiment
    DF_MAP = DF_MAP[DF_MAP.language==CONFIG['run']['src_language']].copy()
    DF_GAP = DF_GAP[DF_GAP.language==CONFIG['run']['src_language']].copy()

    output_rows = []

    # For each model family and hidden state layer combination (n=10)
    for emb in CONFIG['embedding_multiverse']:
        model_family = emb['model_family']
        vector_type = emb['vector_type']
        logger.info(f"{model_family=} {vector_type=}")

        # Filter dataframes to model family and hidden state
        df_map = DF_MAP[
            (DF_MAP.model_version.isin([model_family+'_ia'])) &
            (DF_MAP.vector_type==vector_type)
        ].copy()
        df_gap = DF_GAP[
            (DF_GAP.model_version==model_family+'_ia') &
            (DF_GAP.vector_type==vector_type) 
        ].copy()

        # Deduplicate source words for train-test split
        unique_src_words = sorted(df_map.src_word.unique())

        # Assertions:
        #   Source (src) words: no duplicates in df_map + df_gap
        #   Target (tgt) words: if duplicates, all embedding vectors for same word are equal
        all_src = pd.concat([df_map.src_word, df_gap.src_word])
        assert len(all_src) == all_src.nunique(), \
            'Src words: duplicate src words across map and gap'
        for _, g in df_map.groupby('tgt_word'):
            if len(g) > 1:
                V = np.stack(g.tgt_vec.values)
                assert (V == V[0]).all(), \
                    'Tgt words: non-equal embedding vectors for same tgt word'
        
        # For each of n=100 random train/test splits
        for seed in range(CONFIG['run']['num_iters']):
            logger.info(seed)
            df_map_iter = df_map.copy()

            # Create seed specific train/test split
            # Unique shuffle for source words
            shuffled_src_words = np.random.RandomState(seed).permutation(unique_src_words)
            # N source words to include in Train set (floor if decimal)
            n_train_words = int(len(unique_src_words) * CONFIG['run']['pct_train'])
            # Grab first N shuffled source words for Train set
            train_words = set(shuffled_src_words[:n_train_words])
            # Append assignment to dataframe in column 'split'
            df_map_iter['split'] = df_map_iter.src_word.apply(
                lambda w: 'train' if w in train_words else 'test')
            
            # Sorting for deterministic SVD
            df_map_iter.sort_values(
                ['split', 'src_word', 'tgt_word', 'model_version'], kind='mergesort', inplace=True)

            # For each unique embedding space transformation
            for tf_dm in CONFIG['tf_dm_sets']:
                procrustes_flag=tf_dm['procrustes_flag']
                pca_flag = tf_dm['pca_flag']
                
                # Collect training source and target vectors and create TrainWarehouse
                src_train = np.stack(df_map_iter[df_map_iter.split=='train'].src_vec.values)
                tgt_train = np.stack(df_map_iter[df_map_iter.split=='train'].tgt_vec.values)
                               
                TW = TrainWarehouse(
                    src_vecs=src_train,
                    tgt_vecs=tgt_train,
                    procrustes_flag=procrustes_flag,
                    pca_flag=pca_flag,
                    pca_dim=CONFIG['run']['pca_dim'])
                TW.fit()

                # Deduplicate tgt universe (unique combined train and test target words)
                all_tgt_vecs = np.stack(df_map_iter.tgt_vec.values)
                all_tgt_words = np.array(df_map_iter.tgt_word)

                _, keep_idx = np.unique(all_tgt_words, return_index=True)
                keep_idx = np.sort(keep_idx)
                Y_staging = all_tgt_vecs[keep_idx]
                Y_words = all_tgt_words[keep_idx]          
   
                # Build X staging as concatenation of df_map + df_gap
                X_staging = np.vstack([np.stack(df_map_iter.src_vec), np.stack(df_gap.src_vec)])
                # Save evaluation assignment (train or test from df_map, gap from df_gap) 
                eval_type = np.concatenate([df_map_iter.split.values, ['gap']*len(df_gap)])
                # All words for evaluation
                X_words = np.concatenate([df_map_iter.src_word, df_gap.src_word])
                X_words_translation = np.array(
                    list(df_map_iter.tgt_word.values) + [None]*len(df_gap), dtype=object)   
 
                # Project src (X) and tgt (Y) vectors based on training data
                X = TW.project(X_staging, kind='source')
                Y = TW.project(Y_staging, kind='target')
            
                assert np.isfinite(X).all() and np.isfinite(Y).all(), \
                    'Non-finite values in projected embeddings'
                assert (np.linalg.norm(X, axis=1) > 1e-6).all() \
                    and (np.linalg.norm(Y, axis=1) > 1e-6).all(), \
                    'Near-zero row norm after projection'               
                
                # Compute CS and CSLS pairwise similarity matrices for all source and target words
                sim_all = X @ Y.T
                assert sim_all.shape[0] > CONFIG['run']['csls_k'] \
                    and sim_all.shape[1] > CONFIG['run']['csls_k'], \
                    f"CSLS requires n_src>k and n_tgt>k"
                csls_all = compute_csls(sim_all, CONFIG['run']['csls_k'])
                
                # Run evaluation across all X_words and append assignment per word
                df_eval = pd.DataFrame(
                    run_evaluations(
                        X_words=X_words, 
                        X_words_translation=X_words_translation,
                        Y_words=Y_words,
                        csls_sim=csls_all,
                        cs_sim=sim_all,
                        eval_type=eval_type))
                                
                # Save results to staging output df
                df_eval.insert(0, 'random_seed', seed)
                df_eval.insert(0, 'tf_dm', tf_dm['name'])
                df_eval.insert(0, 'vector_type', vector_type)
                df_eval.insert(0, 'model_family', model_family)
            
                output_rows.append(df_eval)

    # Combine all trial run observations and save
    DF_TRIALS = pd.concat(output_rows, ignore_index=True)    
    save(DF_TRIALS, CONFIG['paths']['pipeline_output'] + '/cg_df_trials.pkl')

    # Compute trial-metric AUCs across all random seeds
    DF_AUC = compute_aucs(DF_TRIALS)
    save(DF_AUC, CONFIG['paths']['pipeline_output'] + '/cg_df_aucs.pkl')


if __name__ == '__main__':
    main()