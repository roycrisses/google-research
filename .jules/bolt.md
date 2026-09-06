## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2026-09-06 - String Deduplication in Loose Evaluation Variants
**Learning:** When deduplicating multi-variant candidates (e.g., candidate response variations in loose evaluation routines), keying off `r.strip()` while storing unstripped `r` drops candidate variants that differ only by leading/trailing whitespace, which can cause false negatives on whitespace-sensitive checkers.
**Action:** Always deduplicate exact candidate strings (`r in seen`) when candidates are evaluated as-is by downstream checkers, and only filter out empty candidate variants using `r.strip()`.
