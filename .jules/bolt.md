## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2024-05-23 - JAX Attention Broadcasting and Reductions
**Learning:** In JAX attention implementations, manually broadcasting projection matrices or 'all-ones' tensors by adding zeros to create 'thick' matrices is a performance anti-pattern. Using native JAX broadcasting in `lax.dot_general` and `jnp.sum` for reductions is more memory-efficient and faster.
**Action:** Replace manual tiling/broadcasting logic with native JAX broadcasting and use built-in reduction functions like `jnp.sum` instead of dot products with ones for better performance and readability.
