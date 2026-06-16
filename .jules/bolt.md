## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-05-21 - ROUGE LCS and Tokenizer Refinements
**Learning:** Micro-optimizations like replacing regex filters with truthiness checks in tokenizers can change the final ROUGE score by including unwanted tokens (e.g., punctuation), violating metric consistency. Also, `itertools` is not part of `six.moves`.
**Action:** Always maintain strict functional parity with original metrics when optimizing evaluation libraries. Use standard library `itertools` directly. Explicitly document performance-critical side effects like row aliasing in DP tables to warn future maintainers about read-only requirements.
