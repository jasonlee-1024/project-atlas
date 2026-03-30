# Project Atlas

A collection of projects and homework assignments from my studies at Columbia University, covering ML systems, LLMs, and related topics. I found many of these assignments genuinely interesting and created this repo to keep them in one place.

## Structure

```
project-atlas/
├── nlp/        # Natural Language Processing
│   ├── hw1/    # BERT and Word2Vec
│   └── hw2/    # GPT-2 from scratch + fine-tuning on 20 Newsgroups
├── hpml/       # High Performance Machine Learning
│   ├── hw1/    # Benchmarking dot product in C, Python, and MKL
│   ├── hw2/    # Profiling Small LLM Workloads
│   └── cuda1/  # CUDA programming assignments
└── hpc/        # High Performance Computing
    ├── hw1/    # Roofline model and Arithmetic Intensity analysis
    └── hw2/    # Matrix multiplication inside a transformer
```

## Highlights
- **NLP HW2**: Implemented GPT-2 from scratch in PyTorch, including KV caching, nucleus sampling, and fine-tuned a classifier on 20 Newsgroups achieving >65% validation accuracy.
- **HPC HW2**: Deep dive into matrix multiplication as a core operation inside transformer architectures.
- **HPML CUDA**: CUDA kernel implementations including matrix multiplication optimizations and unified memory experiments.
