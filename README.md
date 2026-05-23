# Discovering Lexical Gaps Using Embeddings from Multilingual LLMs

### Yoonwon Jung, Aaron S. Cohen, Benjamin K. Bergen
Department of Cognitive Science, University of California San Diego

---

## Overview
This repository houses the code used to run experiments and reproduce the evaluation and figures in the paper, consisting of two different parts. Each part includes its own .md file documenting the hardware used and the compute time. 

1. [Embbedding extraction](./embedding_extraction/)
- A demo based on using one of the bilingual LLMs (`EXAONE-3.5-7.8b`) used in the paper
    - Downloading the model: `download_model_exaone_3.5_7.8b.py` 
    - Getting the embeddings of gap words and storing them in a dictionary format: `gap_embed_exaone_3.5_7.8b.py`
    - Getting the embeddings of non-gap words and storing them in a dictionary format: `non-gap_embed_exaone_3.5_7.8b.py`

2. [Experiments and evaluation](./experiment_pipeline/)
- Processing the embedding dictionaries into dataframes for lexical gap experiment pipeline: `df_creation.py`
- List of configurations for experiment runs: `config.py`
- Experiment pipeline: `pipeline.py`
- Evaluation and visualization: `analysis.ipynb`

Plese refer to the paper for detailed experiment and evaluation procedures.

## Citation
*Discovering Lexical Gaps Using Embeddings from Multilingual LLMs* (pending link)
