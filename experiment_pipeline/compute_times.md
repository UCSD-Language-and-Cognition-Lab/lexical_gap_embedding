### pipeline.py


All estimates computed as average over 5 random seeds, ran locally on 2025 Macbook Air (M4, 16gb)

Embedding dims for (alldims) spaces vary per model family

--

> **Korean->English** (209 non-gap words, 19 gap words)


| Embedding dim | unaligned-alldims | unaligned-pca256 | procrustes-alldims | procrustes-pca256 | Per-seed total |
|---------------|-------------------|------------------|--------------------|-------------------|----------------|
| 1792          | ~8ms              | ~87ms            | ~1.29s             | ~100ms            | ~1.5s          |
| 2560          | ~12ms             | ~90ms            | ~3.8s              | ~115ms            | ~4.0s          |
| 4096          | ~14ms             | ~117ms           | ~15.4s             | ~168ms            | ~15.7s         |



--

> **English->Korean** (235 non-gap words, 27 gap words)


| Embedding dim | unaligned-alldims | unaligned-pca256 | procrustes-alldims | procrustes-pca256 | Per-seed total |
|---------------|-------------------|------------------|--------------------|-------------------|----------------|
| 1792          | ~10ms             | ~90ms            | ~1.17s             | ~97ms             | ~1.4s          |
| 2560          | ~14ms             | ~96ms            | ~3.6s              | ~125ms            | ~3.8s          |
| 4096          | ~16ms             | ~121ms           | ~14.2s             | ~148ms            | ~14.5s         |


--

### embedding_retrieval.py

