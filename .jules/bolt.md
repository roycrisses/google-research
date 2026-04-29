## 2025-05-15 - [ROUGE LCS Optimization]
**Learning:** In the `rouge-score` library, LCS (Longest Common Subsequence) calculation was the primary bottleneck due to its O(M*N) complexity implemented with nested loops and expensive table allocations. In Python, these overheads are significant.
Two key patterns emerged:
1. Heuristic Early-Exit: Using `set` intersection to detect disjoint token sequences allows skipping the expensive DP table calculation entirely. This is highly effective for `rougeLsum` where many sentence pairs have no overlap.
2. List Operations: `list.insert(0, x)` is O(N) in Python. Replacing it with `list.append(x)` followed by `list.reverse()` in the LCS backtracking step provides a measurable boost.
3. Space Complexity: For `rougeL`, only the LCS length is needed, so O(min(M, N)) space DP is sufficient and faster due to fewer allocations.

**Action:** Always check for disjoint sets before running O(N^2) overlap algorithms. Prefer `append` + `reverse` over `insert(0)`. Minimize allocations in inner loops by reusing rows or using space-optimized DP where full tables aren't needed.
