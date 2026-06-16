## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2026-05-21 - ROUGE LCS and N-Gram Optimizations
**Learning:** In Python, LCS performance can be improved by ~60-90% by filtering input sequences to their intersection before running DP (problem size reduction) and using row-sharing in the full table to skip tokens missing from the candidate set. N-gram creation is also much faster using `itertools.islice` and `zip` than manual slicing in a generator.
**Action:** Always filter LCS inputs to their common intersection when possible. Use `lru_cache` for sentence-pair LCS in multi-document evaluation.
