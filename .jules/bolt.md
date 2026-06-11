## 2025-05-22 - ROUGE LCS Algorithmic Optimizations
**Learning:** The default ROUGE LCS implementation suffered from several performance anti-patterns in Python: O(M*N) memory usage for simple length checks, O(N^2) list building using `insert(0, ...)`, and redundant O(M*N) DP calculations for disjoint token sequences or non-overlapping sentences in summaries.
**Action:** Always use space-optimized DP ($O(\min(M, N))$) when only the length is needed. Use `append()` + `reverse()` for efficient list building. Implement fast-path checks using `set` intersections to bypass expensive algorithms. Pre-calculate sets in loops to avoid redundant conversions. Use local variable lookups and conditional expressions instead of `max()` in tight loops.

## 2025-06-11 - JAX Einsum vs Matmul Optimization
**Learning:** In JAX,  is a general-purpose tool that can sometimes result in suboptimal lowering to XLA HLO compared to  for standard matrix multiplications. Specifically, for 3D batch matrix multiplications like "bij,jk,ni->bnk", nested  calls can be significantly faster (up to ~2.2x speedup observed on CPU) as they more directly map to optimized BLAS/XLA Dot kernels.
**Action:** Prefer  over  for standard 2D and 3D matrix multiplications when performance is critical. Always verify that broadcasting behavior and dimensions are correctly handled when nesting .

## 2025-06-11 - JAX Einsum vs Matmul Optimization
**Learning:** In JAX, `jnp.einsum` is a general-purpose tool that can sometimes result in suboptimal lowering to XLA HLO compared to `jnp.matmul` for standard matrix multiplications. Specifically, for 3D batch matrix multiplications like "bij,jk,ni->bnk", nested `jnp.matmul` calls can be significantly faster (up to ~2.2x speedup observed on CPU) as they more directly map to optimized BLAS/XLA Dot kernels.
**Action:** Prefer `jnp.matmul` over `jnp.einsum` for standard 2D and 3D matrix multiplications when performance is critical. Always verify that broadcasting behavior and dimensions are correctly handled when nesting `matmul`.
