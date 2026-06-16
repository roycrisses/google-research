## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-05-23 - Pythonic N-Gram and Tokenization Gains
**Learning:** High-frequency utility functions like `tokenize` and `_create_ngrams` often contain hidden overhead from redundant regex matches and intermediate list slicing. Using `itertools.islice` with `six.moves.zip` for n-grams and simple truthiness checks for tokens (when constraints are guaranteed) provides massive speedups.
**Action:** Prioritize iterator-based sequence processing to minimize memory pressure. Use `set` membership to skip entire DP rows in LCS when no match is possible, effectively reusing previous row results.
