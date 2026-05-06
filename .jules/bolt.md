## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-05-23 - Regex Bottleneck in Tokenization
**Learning:** Using `regex.match()` in a tight loop during tokenization (e.g., filtering thousands of tokens) can be surprisingly expensive in Python. When tokens are already processed (e.g., non-alphanumeric replaced by spaces), a simple truthiness check `if x` is significantly faster and logically equivalent to an alphanumeric regex check.
**Action:** Avoid regex for simple string validation if prior processing guarantees the string's format. Use truthiness or basic string methods for filtering in performance-critical paths.
