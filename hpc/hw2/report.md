# HW 2

## Problem 1

### Matrix Multiplication 1:  Projection

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

#### (a) Dimensions

- Input
  - X : (B, S, d)
  - $W^Q$, $W^Q$, $W^V$ : (d, $d_k$)
- Output
  - Q: (B, S, $d_k$)

$d_k = d/h$, h is number of attention heads.

#### (b) Conceptual Explanation

Conceptually, $Q = XW^Q$ is a **linear projection**. Each token in $X$ is a $d$-dimensional vector containing all its semantic information. Multiplying by $W^Q$ projects it down to a $d_k$-dimensional subspace, extracting only the features relevant to the "query" role — essentially, "what information is this token looking for?"

Similarly, $W^K$ extracts "what information can this token offer?" and $W^V$ extracts "what is this token's actual content?" The same input $X$ is projected into three different semantic roles through three different learned weight matrices, preparing them for the subsequent attention score computation.

#### (c) Cost at Large Scale

The matrix multiplication $Q = XW^Q$ has a computational complexity of $O(B \cdot S \cdot d \cdot d_k)$. At large scale, this becomes expensive because:

1. **Sequence length $S$**: Modern LLMs use long contexts (e.g., $S$ = 100K+). Since this multiplication is performed for every token in the sequence, cost grows linearly with $S$.
2. **Hidden dimension $d$**: Larger models use wider hidden dimensions (e.g., $d$ = 12288 for GPT-3). The weight matrix $W^Q$ is $(d, d_k)$, so larger $d$ directly increases both computation and memory.
3. **Repeated computation**: This projection is not done once — it is done three times ($W^Q$, $W^K$, $W^V$) per attention layer, across every layer in the model (e.g., 96 layers in GPT-3). The total cost multiplies quickly.

### Matrix Multiplication 2: Attention Score

$$A = QK^T$$

#### (a) Dimensions

- **Input** $Q$: $(B, S, d_k)$
- **Input** $K^T$: $(B, d_k, S)$
- **Output** $A$: $(B, S, S)$

#### (b) Conceptual Explanation

This computes the dot product between every pair of tokens, producing an $S \times S$ similarity matrix. Each entry $(i, j)$ represents how much token $i$ should attend to token $j$ — essentially measuring "how relevant is token $j$ to what token $i$ is looking for?" After scaling by $\frac{1}{\sqrt{d_k}}$ and applying softmax row-wise, this becomes a probability distribution over all tokens for each query position.

#### (c) Cost at Large Scale

The computational complexity is $O(B \cdot S^2 \cdot d_k)$. The critical factor is the **quadratic dependence on $S$** — doubling the sequence length quadruples both computation and memory. The output matrix $(S, S)$ must be stored in GPU memory, which becomes prohibitive for long sequences (e.g., $S = 100K$ produces a $10^{10}$-element matrix). This $S^2$ bottleneck is the primary motivation behind efficient attention methods like Flash Attention, which uses tiling to avoid materializing the full $(S, S)$ matrix in memory.

---

## Problem 2

### (a) Total Number of Floating-Point Operations

operation numbers = $m \times n \times 2k$

### (b) Evaluation at $m = n = k = 4096$

operation numbers = $m \times n \times 2k = 137,438,953,472$

### (c) Scaling When All Dimensions Are Doubled

If all three dimensions are doubled, the total work is multiplied by $2^3$.

### (d) Importance in Large Language Models

LLMs are dominated by matrix multiplications at every attention and
feed-forward layer. Because flop count scales as $2mnk$, model size
has an outsized effect on compute cost.

In GPT-3 for example, the hidden dimension is $d = 12{,}288$ and
context length is 2,048 tokens. Each linear projection multiplies
matrices of size $(2048 \times 12288)$ by $(12288 \times 12288)$,
costing $\sim 6 \times 10^{11}$ flops — **per layer**. With 96
layers, the total compute per forward pass is enormous.

The **scaling laws** (Kaplan et al., 2020) show empirically that loss
decreases predictably as $N$ (parameters) and $D$ (training data)
grow. Since $N \sim d^2$ and flops $\sim d^2$ per token, **doubling
the hidden dimension quadruples the compute** — making the $2mnk$
scaling the central bottleneck in training and deploying large models.

---

## Problem 3

### (a) Arithmetic Intensity

Arithmetic intensity is defined as flops per byte of memory access.

**GEMV** ($y = Ax$, $A \in \mathbb{R}^{m \times n}$):

- Flops: $2mn$
- Memory: $mn + n + m \approx mn$ bytes
- Arithmetic intensity: $\dfrac{2mn}{mn + n + m} \approx 2$

**GEMM** ($C = AB$, $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$):

- Flops: $2mnk$
- Memory: $mk + kn + mn$ bytes
- Arithmetic intensity: $\dfrac{2mnk}{mk + kn + mn} $

GEMM's arithmetic intensity grows with matrix size, while GEMV's stays
constant at $\approx 2$. Modern GPUs (e.g. A100) have a compute ceiling
of ~312 TFLOPS but a memory bandwidth ceiling of ~2 TB/s — only
operations with high arithmetic intensity can escape the memory
bottleneck and approach peak compute.

### (b) Data Reuse

In GEMM, each column of $A$ is reused across all $n$ columns of $B$,
and each row of $B$ is reused across all $m$ rows of $A$. This allows
tiles of $A$ and $B$ to be loaded into fast on-chip shared memory
(SRAM) and reused many times before being evicted — this is the core
idea behind **tiled matrix multiplication**.

In GEMV, each row of $A$ is used exactly **once** to compute a single
scalar output $y_i$. There is no opportunity to reuse $A$ across
multiple right-hand sides, so every element must be streamed from
global memory and then discarded.

### (c) Memory Bandwidth

GEMM produces an $m \times n$ output matrix, giving $O(mn)$ independent
output elements that can be computed in parallel. This is sufficient to
saturate thousands of GPU cores and hide memory latency through
**warp-level parallelism**.

GEMV produces only $m$ output scalars. For moderate $m$, this is often
insufficient to fully occupy the GPU, leaving many cores idle and
reducing effective throughput.

---

## Problem 4

### (a) Why Batching Improves Throughput

1. **Parallel Computation**: A GPU has thousands of cores. A single sequence doesn't provide enough work to keep them all busy. Batching 64 sequences gives the GPU enough parallel work to saturate its compute units.
2. **Data Reuse**: The model weights are loaded from GPU memory once and reused across all 64 sequences in the batch, amortizing the expensive memory access cost. Without batching, the same weights must be loaded repeatedly for each sequence.

Together, these two factors shift the bottleneck from **memory-bound** (waiting on data) to **compute-bound** (fully utilizing the GPU), dramatically improving throughput.

### (b) Two Tradeoffs of Increasing Batch Size

1. **Latency vs. throughput.** Larger batches improve throughput

(requests per second) but individual requests must wait longer to
be returned — unacceptable in latency-sensitive applications.

1. **Memory usage.** Each request requires storing its own KV cache

in GPU VRAM. Large batches can exhaust available memory, limiting
maximum sequence length or forcing slower CPU offloading.

### (c) Why the Largest Batch Size Is Not Always Best

If the system receives requests at a low arrival rate, waiting to
fill a batch of 64 before processing introduces unnecessary queuing
delay. A smaller batch processed immediately can deliver lower latency
with only a modest reduction in throughput — making it the better
choice under light or bursty traffic conditions.

## Problem 5

### (a) Compute-Bound or Memory-Bound?

The GPU's **operational intensity threshold** (ridge point) is:

Ridge Point = Peak Compute / Memory Bandwidth = 40 TFLOP/s / 1.5 TB/s ≈ 26.67 flops/byte

The kernel's arithmetic intensity is **50 flops/byte > 26.67 flops/byte**, so the kernel is **compute-bound**.

### (b) Best-Case Runtime

Since the kernel is compute-bound, the limiting factor is peak compute:

Runtime = Total FLOPs / Peak Compute = 1.0 × 10¹² / 40 × 10¹² = 0.025 s (25 ms)

### (c) Optimizations

1. **Software — Tiling (Shared Memory Blocking)**: Break the matrix into tiles that fit in GPU shared memory / registers to maximize data reuse and reduce global memory traffic.
2. **Algorithmic — Reduced Precision (FP16 / INT8)**: Use lower-precision arithmetic (e.g., Tensor Cores with FP16), which effectively doubles or quadruples peak TFLOP/s, reducing runtime proportionally.

