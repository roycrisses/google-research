## 2025-05-14 - LCS Optimization in ROUGE
**Learning:** Python's `max()` function and list `insert(0, ...)` in the tight loop of LCS (Longest Common Subsequence) are major bottlenecks. O(M*N) space allocation for the DP table is also expensive for large texts. Disjoint token sets are common in summary evaluation and can be used for a fast early-exit.
**Action:** Use `v1 if v1 >= v2 else v2` instead of `max(v1, v2)`. Use `append()` and `reverse()` for O(1) amortized list building. Use O(min(M, N)) space for LCS length. Pre-calculate token sets to skip DP for disjoint sequences.
