## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-05-23 - Vectorizing Date Range Search with searchsorted
**Learning:** Calling `.apply()` row-by-row on a pandas DatetimeIndex with operations like `holiday.dates(dt - delta, dt + delta)` causes $O(N \cdot M)$ redundant date lookups, leading to extreme bottlenecks (e.g., >500 seconds for ~26k time steps across 18 holidays).
**Action:** Pre-generate all occurrence dates for the entire dataset time range once per holiday, and use `np.searchsorted` on `self.dti - delta` to vectorially look up closest occurrences in $O(N \log M)$ time, reducing feature extraction time from ~500s to ~0.35s (~1,500x speedup).
