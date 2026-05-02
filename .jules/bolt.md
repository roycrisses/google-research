## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-05-02 - LCS Loop and Cache Optimizations
**Learning:** Python list indexing in tight loops (like LCS DP) adds significant overhead. Caching results for sentence pairs is crucial for summary-level ROUGE where sentences are often compared multiple times or across different evaluations.
**Action:** Use `enumerate(seq, 1)` instead of `range(len(seq))` and indexing in tight loops. Implement `functools.lru_cache` for expensive algorithmic functions, converting lists to tuples for hashability.
