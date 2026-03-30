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

- **NLP HW1**: Word2Vec embeddings and BERT fine-tuning experiments.
- **NLP HW2**: Implemented GPT-2 from scratch in PyTorch, including KV caching, nucleus sampling, and fine-tuned a classifier on 20 Newsgroups achieving >65% validation accuracy.
- **HPC HW1**: Roofline model analysis and Arithmetic Intensity measurements to understand hardware performance bottlenecks.
- **HPC HW2**: Deep dive into matrix multiplication as a core operation inside transformer architectures.
- **HPML HW1**: Benchmarking dot product performance across C, Python, and Intel MKL to understand low-level compute efficiency.
- **HPML HW2**: Profiling small LLM workloads to identify performance bottlenecks.
- **HPML CUDA**: CUDA kernel implementations including matrix multiplication optimizations and unified memory experiments.
