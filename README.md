## The Repository for the EMNLP 2025 Paper ***The Medium is Not the Message: Deconfounding Document Embeddings via Linear Concept Erasure*** 

[![arXiv](https://img.shields.io/badge/arXiv-2507.01234-b31b1b)](https://arxiv.org/abs/2507.01234)
[![license](https://img.shields.io/github/license/y-fn/deconfounding-embeddings?label=License)](https://github.com/y-fn/deconfounding-embeddings/blob/main/LICENSE)

## Introduction

Embedding-based similarity metrics between text sequences can be affected not only by the content dimensions of interest but also by spurious attributes such as source or language. These document-level confounders pose challenges for many applications, particularly those combining texts from different corpora. We demonstrate that a debiasing algorithm removing information about observed confounders from encoder representations significantly improves similarity and clustering metrics across tasks without degrading out-of-distribution performance.

## Erasure Pipeline

The current [pipeline](https://github.com/y-fn/deconfounding-embeddings/blob/main/style_erasure_pipeline.py) generates and saves plots for the following visualizations:
- K-means clusterings, both before and after LEACE erasure;
- PCA projections, before and after LEACE erasure;
- Top-k retrieval results, before and after erasure;
- Total # of exact pairs, before and after erasure.

## Data

In addition, we provide all [data](https://github.com/y-fn/deconfounding-embeddings/tree/main/data) used in our paper.

## Citation

If you find our work helpful, please consider citing us: 
```shell
@article{fan2025medium,
  title={The Medium Is Not the Message: Deconfounding Document Embeddings via Linear Concept Erasure},
  author={Fan, Yu and Tian, Yang and Ravfogel, Shauli and Sachan, Mrinmaya and Ash, Elliott and Hoyle, Alexander},
  journal={arXiv preprint arXiv:2507.01234},
  year={2025}
}
```
