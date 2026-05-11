## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-05-11 - JAX Performer Broadcasting Optimizations
**Learning:** Manually broadcasting projection matrices or "all-ones" tensors by adding zeros to create "thick" matrices is a performance anti-pattern in JAX. It leads to unnecessary memory allocation and slower execution.
**Action:** Use native JAX broadcasting in `lax.dot_general` and `jnp.sum`. For example, a projection can be done with `lax.dot_general(data, projection_matrix, (((data.ndim - 1,), (1,)), ((), ())))`. Replacing manual broadcast and `lax.dot_general` with `jnp.sum` and broadcasting improved bidirectional attention performance by ~35%.
