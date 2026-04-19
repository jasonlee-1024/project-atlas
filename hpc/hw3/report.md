# Report

## Problem 1

### (a)
After rank 0 and rank 1 execute send function, they may both wait for other ranks to receive contents. Therefore, the program may deadlock.

### (b)
- Change the order in rank 1; execute Recv first then execute Send function.

```c
if (rank == 0) {
    MPI_Send(&a, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(&b, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
} else {  // rank == 1
    MPI_Recv(&b, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(&a, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
}
```

- Use non-blocking `MPI_Isend` so neither process blocks waiting for the other.

```c
MPI_Request req;
MPI_Isend(&a, 1, MPI_INT, 1-rank, 0, MPI_COMM_WORLD, &req);
MPI_Recv(&b, 1, MPI_INT, 1-rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Wait(&req, MPI_STATUS_IGNORE);
```

### (c)
- When the variable is small enough to fit in MPI's internal buffer (eager protocol), the send function returns immediately without waiting for the receiver. In that situation, the original code may appear to work in practice.

## Problem 2

### (a)
A tree-based reduction runs in $\log_2 p$ rounds. In each round, all communications happen in parallel, so the cost per round is one message: $\alpha + \beta m$. Total cost:

$$T = \log_2 p \cdot (\alpha + \beta m)$$

### (b)
Each process contributes exactly one double, so $m = 8$ bytes:

$$T = \log_2 p \cdot (\alpha + 8\beta)$$

### (c)
In the naive strategy, rank 0 receives one message from each of the other $p-1$ ranks sequentially. Total cost:

$$T_{\text{naive}} = (p-1) \cdot (\alpha + 8\beta)$$

Compared to tree reduction, the naive approach is $O(p)$ vs $O(\log p)$. For example, at $p = 1024$: tree reduction takes 10 rounds while the naive approach takes 1023 rounds — roughly 100× slower.

## Problem 3
### (a)
We can use broadcast algorithm for this operation.
- First, broadcast the full vector $x$ to each process.
- Then, each process runs local matrix-vector multiplication independently.


### (b)
- The full vector $x$: the complete $x$ must be broadcast via `MPI_Bcast` to all processes before the local matvec can proceed.

### (c)
When doing local matrix-vector multiplication, each process holds a submatrix $A_{\text{local}} \in \mathbb{R}^{(n/p) \times n}$. The computational cost per process is:

$$\text{cost} = \frac{n}{p} \times (2n - 1) \approx \frac{2n^2}{p} \text{ flops}$$

### (d)
Assume all data is double-precision (8 bytes per element).

The dominant communication cost is broadcasting the full vector $x$ (size $8n$ bytes) to all processes using a tree-based `MPI_Bcast` with $\log_2 p$ rounds:

$$T_{\text{bcast}} = \log_2 p \cdot (\alpha + 8\beta n)$$

## Problem 4

### (a)
Before the iteration loop, broadcast the initial $x$ to all processes:
1. `MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD)`

In each iteration:
1. Each process computes its local residual and update: $x^{(k+1)}_{\text{local}} = x^{(k)}_{\text{local}} + \omega(b_{\text{local}} - A_{\text{local}} x^{(k)})$
2. Execute `MPI_Allgather` to assemble the full updated $x^{(k+1)}$ on all processes.

### (b)
The main collective communication per iteration is **`MPI_Allgather`** on the iterate $x$ — each process contributes its updated $n/p$ elements and receives the full $n$-element vector.

### (c)
Even with perfectly balanced compute work $O(n^2/p)$ per process, every iteration requires an Allgather of the full vector $x$. Using a ring-based Allgather, the cost is:

$$T_{\text{allgather}} = (p-1)\left(\alpha + \beta \cdot \frac{8n}{p}\right) \approx p\alpha + 8\beta n$$

The $p\alpha$ term grows linearly with $p$, while the compute work shrinks as $O(n^2/p)$. As $p$ increases, communication eventually dominates and becomes the bottleneck regardless of how well the flops are balanced.

### (d)
**Strong scaling:** Fix the problem size $n$ and increase $p$. Ideally, doubling $p$ halves the runtime. In practice, each process's compute work $O(n^2/p)$ shrinks, but the Allgather cost $p\alpha + 8\beta n$ grows with $p$. Eventually communication dominates and adding more processes no longer helps.

**Weak scaling:** Keep the per-process workload fixed by increasing $n$ proportionally with $p$ (i.e., $n/p$ constant). Ideally the runtime stays constant. In practice, the $p\alpha$ term in the Allgather cost still grows with $p$, so efficiency degrades as $p$ increases.

## Problem 5

### (a)
1. Increasing the number of MPI processes increases communication overhead (e.g., more rounds in collective operations, higher latency cost). Beyond a certain point, the communication cost outweighs the benefit of parallelism and overall performance gets worse.
2. If the number of MPI processes exceeds the problem size, some processes will have no work and remain idle, wasting resources without any speedup.

### (b)
- `MPI_Barrier`: ensures all processes have reached the same point before any can proceed. No process exits the barrier until every process has entered it.
- `MPI_Allreduce`: combines a reduction (e.g., global sum) with a broadcast of the result. All processes must participate and synchronize before any process receives the final result.

In both cases, the wall-clock time is gated by the slowest process — all others must wait.

### (c)
Load imbalance means some processes have significantly more work than others. Since collective operations act as implicit synchronization barriers, the faster processes finish early and sit idle waiting for the slowest one. The overall wall-clock time is determined by the maximum load across all processes, not the average.


