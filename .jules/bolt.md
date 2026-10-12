## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-05-23 - SciPy vs NumPy Sampling & Simplex Vectorization
**Learning:** `scipy.stats.dirichlet.rvs` introduces significant Python object wrapper overhead compared to `np.random.dirichlet`. Furthermore, sampling from the fundamental simplex can be vectorized using `np.cumsum` over reverse-indexed Dirichlet weights instead of constructing a $(d+1) \times d$ matrix and elementwise multiplying.
**Action:** Prefer `np.random.dirichlet` over `scipy.stats.dirichlet.rvs`. Convert simplex linear combinations of indicator vectors to `np.cumsum` suffix/prefix sums to eliminate matrix allocations in sampling loops.
