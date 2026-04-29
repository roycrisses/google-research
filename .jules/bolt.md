## 2025-04-29 - LCS Performance Bottleneck in ROUGE
**Learning:** The ROUGE LCS (Longest Common Subsequence) implementation had an O(M*N) bottleneck in table construction and an O(L^2) bottleneck in backtracking due to list.insert(0, x). Furthermore, summary-level LCS was redundantly creating sets.
**Action:**
1. Use `set.isdisjoint()` as an early-exit for sequences with no common tokens to skip the O(M*N) DP table.
2. Replace `list.insert(0, x)` with `list.append(x)` followed by `list.reverse()` to ensure O(L) backtracking.
3. Pre-calculate sets for candidate sentences in summary-level LCS to avoid redundant work in nested loops.
