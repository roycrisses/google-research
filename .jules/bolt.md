## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-05-23 - Python Loop Reductions vs NumPy Overhead
**Learning:** Calling `np.mean` thousands of times inside nested Python loops for tiny arrays (e.g. 1-5 elements) incurs severe C-extension dispatch and NumPy array creation overhead (~10–15 µs per call). In pure Python, `sum(lst) / len(lst)` is ~100x faster per calculation (~0.1 µs).
**Action:** Use pure Python reductions (`sum(...) / len(...)`) inside tight loops over small Python lists, and only convert to NumPy / call `np.mean` once on the aggregated outer list of results.

## 2025-05-24 - Sobolev-Chebyshev Polynomial Evaluation
**Learning:** Recursive polynomial evaluation algorithms that call lower-degree orthogonal polynomials (like Chebyshev polynomials) independently cause $O(n^2)$ redundant evaluations across recursive calls.
**Action:** Replace recursive polynomial evaluations with a single $O(n)$ bottom-up iterative loop pass, precomputing all lower-degree Chebyshev values ($T_k(s_0), T_k(s_x)$) and computing $S_k(x)$ iteratively.
