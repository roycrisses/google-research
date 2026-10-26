## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-05-23 - Python Loop Reductions vs NumPy Overhead
**Learning:** Calling `np.mean` thousands of times inside nested Python loops for tiny arrays (e.g. 1-5 elements) incurs severe C-extension dispatch and NumPy array creation overhead (~10–15 µs per call). In pure Python, `sum(lst) / len(lst)` is ~100x faster per calculation (~0.1 µs).
**Action:** Use pure Python reductions (`sum(...) / len(...)`) inside tight loops over small Python lists, and only convert to NumPy / call `np.mean` once on the aggregated outer list of results.

## 2025-05-24 - Avoiding O(N^2) Matrix Allocation for Diagonal Extraction
**Learning:** Extracting the diagonal of a Gram/covariance matrix via `jnp.diag(jnp.matmul(x, x.T))` creates an unnecessary $O(N^2)$ matrix intermediate taking $O(N^2 \cdot D)$ FLOPs and $O(N^2)$ memory space (e.g. 400 MB for $N=10,000$), risking OOM errors on large datasets.
**Action:** Compute row/vector squared norms directly with `jnp.sum(jnp.square(x), axis=-1)` to achieve $O(N \cdot D)$ time and $O(N)$ memory footprint.
