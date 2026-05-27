## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-05-27 - JAX Broadcasting and Reduction Anti-patterns
**Learning:** Manually broadcasting projection matrices or 'all-ones' tensors by adding zeros/ones to create 'thick' matrices in JAX is a performance anti-pattern. It leads to unnecessary memory allocation and slower execution. Native JAX broadcasting in `lax.dot_general` and using `jnp.sum` for contractions with all-ones vectors are significantly more efficient.
**Action:** Always prefer implicit broadcasting via `dimension_numbers` in `lax.dot_general` and built-in reduction functions like `jnp.sum` over explicit matrix materialization when working with JAX/Flax.
