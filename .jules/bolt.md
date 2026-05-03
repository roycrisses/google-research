## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-05-22 - Fast-path Membership Skips in DP Loops
**Learning:** In Python LCS implementations, performing an O(1) membership check before the inner DP loop can skip the entire O(N) inner loop if the current token has no matches. In `_lcs_table`, assigning the current row as a reference to the previous row (`table[i] = table[i-1]`) during a skip is a zero-cost way to maintain correctness for backtracking.
**Action:** Always pre-calculate a `set` for sequences in DP algorithms to enable row-skipping.
