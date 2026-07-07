
import unittest
import jax
import jax.numpy as jnp
from cmmd import distance

class DistanceTest(unittest.TestCase):

  def test_mmd_correctness(self):
    # Use a fixed seed for reproducibility.
    key = jax.random.PRNGKey(42)
    n, d = 10, 5
    x = jax.random.normal(key, (n, d))
    y = jax.random.normal(key, (n, d))

    # Manually compute MMD for verification.
    def manual_mmd(x, y):
      sigma = 10
      scale = 1000
      gamma = 1 / (2 * sigma**2)

      def rbf_kernel(a, b):
        sqdist = jnp.sum((a - b)**2)
        return jnp.exp(-gamma * sqdist)

      k_xx = 0
      for i in range(n):
        for j in range(n):
          k_xx += rbf_kernel(x[i], x[j])
      k_xx /= (n * n)

      k_yy = 0
      for i in range(n):
        for j in range(n):
          k_yy += rbf_kernel(y[i], y[j])
      k_yy /= (n * n)

      k_xy = 0
      for i in range(n):
        for j in range(n):
          k_xy += rbf_kernel(x[i], y[j])
      k_xy /= (n * n)

      return scale * (k_xx + k_yy - 2 * k_xy)

    expected = manual_mmd(x, y)
    actual = distance.mmd(x, y)

    # Using delta instead of places because actual/expected are small.
    self.assertAlmostEqual(float(actual), float(expected), delta=1e-4)

if __name__ == '__main__':
  unittest.main()
