## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-05-18 - JAX Fast Attention Broadcasting and Sums
**Learning:** Manually broadcasting projection matrices or 'all-ones' tensors (creating 'thick' matrices) in JAX is a performance anti-pattern that wastes memory and computation. Native JAX broadcasting in `lax.dot_general` and `jnp.sum` is more efficient. Also, deleting variables that seem redundant in one branch (like `index`) can cause NameErrors in other branches (like unidirectional attention).
**Action:** Leverage native JAX broadcasting and reduction operations. Always verify all functional paths (bidirectional AND unidirectional) when refactoring shared code.
