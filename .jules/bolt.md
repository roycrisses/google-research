## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2026-05-17 - ROUGE Tokenization and N-Gram Optimizations
**Learning:** Python regex matching in a loop for token filtering (`VALID_TOKEN_RE.match(x)`) is significantly slower than a simple truthiness check (`if x`) when the input characters are already constrained by previous steps. N-gram creation using generator expressions with slicing is less efficient than using `itertools.islice` with `zip`.
**Action:** Replace regex-based filtering with simple truthiness checks if safe. Use `collections.Counter(zip(*(itertools.islice(tokens, i, None) for i in range(n))))` for efficient n-gram generation.
