### Compute time for extracting embeddings

All estimates computed in a remote server with 2 NVIDIA RTX A6000 (48GB VRAM) GPUs available, single GPU used
See environment_embed_extraction.yml for the full list of dependencies

---

> **Non-Gap words**   
> based on running .py files equivalent to `non-gap_embed_exaone_3.5_7.8b.py` for each model
- first layer, last layer, and the average of last4 layers were extracted
- sentence embeddings are extracted from kanana-nano, in addition to other layers

| Model | Compute time |
|---|---|
| EXAONE-3.5-2.4B-Instruct | ~26.7s |
| EXAONE-3.5-7.8B-Instruct | ~224.2s |
| Kanana-nano | ~48.0s |
| Kanana-1.5-2.1b-base | ~25.4s |
| Kanana-1.5-8b-base | ~291.2s |

---

> **Gap words**  
> based on running .py files equivalent to `gap_embed_exaone_3.5_7.8b.py` for each model
- first layer, last layer, and the average of last4 layers were extracted
- sentence embeddings are extracted from kanana-nano, in addition to other layers

| Model | Compute time |
|---|---|
| EXAONE-3.5-2.4B-Instruct | ~7.6s |
| EXAONE-3.5-7.8B-Instruct | ~25.1s |
| Kanana-nano | ~7.8s |
| Kanana-1.5-2.1b-base | ~7.3s |
| Kanana-1.5-8b-base | ~24.2s |