Experiment scripts (df_creation.py, pipeline.py) use .pkl for local load and save.

Data files on github (df_map, df_gap) are stored as .parquet

experiment_pipeline.utils lists_to_vecs function restores src/tgt vectors in dataframe back to 
np.float32 arrays, which can be saved locally as .pkl for pipeline input and output.

    df_map = lists_to_vecs(pd.read_parquet('cg_df_map.parquet'), ['src_vec', 'tgt_vec'])
    df_gap = lists_to_vecs(pd.read_parquet('cg_df_gap.parquet'), ['src_vec'])
    save(df_map, 'data/cg_df_map.pkl')
    save(df_gap, 'data/cg_df_gap.pkl')
