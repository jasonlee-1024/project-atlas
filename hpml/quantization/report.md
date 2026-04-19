# Report

Author: Shen li (sl6008)


## Q6: Pruning and Fine-Tuning

1. Higher sparsity levels lead to lower accuracy after fine-tuning.

2. At low sparsity levels (30–50%), fine-tuning fully recovers accuracy and can even exceed the original baseline, likely because removing small-magnitude weights acts as regularization. As sparsity increases to 70–90%, accuracy recovery after fine-tuning diminishes significantly, since too few non-zero weights remain to compensate for the lost model capacity.

## Q7: Pruning Before vs. After Training

1. Pruning before training produces higher accuracy than pruning after training because the network gets to train from scratch within the sparse structure, allowing all remaining weights to fully adapt. When pruning is applied after training, it removes weights from an already-optimized configuration without any recovery, causing an immediate accuracy drop that cannot be compensated without fine-tuning.

2. This result suggests that learned weight magnitudes are not a reliable indicator of which weights should be pruned. Small weights after training are not necessarily unimportant — they may play a role in the overall optimized solution.

## Q8: GPTQ

**1. Main innovations:**

GPTQ introduces three innovations over OBQ. 
- First, the Arbitrary Order Insight quantizes all rows in the same fixed column order rather than each row's own greedy order, so the shared Hessian needs less computation.
- Second, Lazy Batch-Updates defer global weight updates until a block of $B=128$ columns is fully processed, resolving the memory-bandwidth bottleneck on GPU.
- Third, Cholesky Reformulation precomputes all required rows of $\mathbf{H}^{-1}$ using a numerically stable Cholesky decomposition with mild dampening, preventing indefiniteness on very large models.

**2. Scalability and second-order information:**

- Prior methods like OBQ quantize each row in its own greedy order, requiring the Hessian inverse to be recomputed per weight — this becomes prohibitively expensive at billion-parameter scale. GPTQ quantizes all rows in the same fixed column order, so the Hessian inverse is shared across rows and updated only once per column, cutting runtime by orders of magnitude.
- GPTQ minimizes only the local per-layer reconstruction error rather than the global loss, and estimates the Hessian from just 128 calibration samples rather than the full dataset — achieving only 0.03 perplexity loss on OPT-175B at 4-bit vs. RTN's 2.2-point drop.

**3. Limitation and future work:**

- A key limitation of GPTQ is that its speedups come only from reduced memory movement, not from actual computational reductions — the quantized model still performs the same number of operations. - Additionally, the method does not consider activation quantization, which limits the overall compression benefit. 
- We may make it achievable through carefully-designed GPU kernels and existing complementary techniques.
