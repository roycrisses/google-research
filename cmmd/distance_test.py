
import unittest
import jax
import jax.numpy as jnp
from cmmd import distance

class TestMMD(unittest.TestCase):
    def test_mmd_result_consistency(self):
        # Set seeds for reproducibility
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        n, d = 10, 5
        x = jax.random.normal(k1, (n, d))
        y = jax.random.normal(k2, (n, d))

        # Original implementation style for reference
        def original_mmd(x, y):
            x = jnp.asarray(x)
            y = jnp.asarray(y)
            x_sqnorms = jnp.diag(jnp.matmul(x, x.T))
            y_sqnorms = jnp.diag(jnp.matmul(y, y.T))
            gamma = 1 / (2 * 10**2)
            k_xx = jnp.mean(jnp.exp(-gamma * (-2 * jnp.matmul(x, x.T) + jnp.expand_dims(x_sqnorms, 1) + jnp.expand_dims(x_sqnorms, 0))))
            k_xy = jnp.mean(jnp.exp(-gamma * (-2 * jnp.matmul(x, y.T) + jnp.expand_dims(x_sqnorms, 1) + jnp.expand_dims(y_sqnorms, 0))))
            k_yy = jnp.mean(jnp.exp(-gamma * (-2 * jnp.matmul(y, y.T) + jnp.expand_dims(y_sqnorms, 1) + jnp.expand_dims(y_sqnorms, 0))))
            return 1000 * (k_xx + k_yy - 2 * k_xy)

        expected = original_mmd(x, y)
        actual = distance.mmd(x, y)

        np_expected = float(expected)
        np_actual = float(actual)

        self.assertAlmostEqual(np_expected, np_actual, places=4)

    def test_mmd_zero_distance(self):
        # Identical sets should have zero distance
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (10, 5))
        dist = distance.mmd(x, x)
        self.assertAlmostEqual(float(dist), 0.0, places=4)

if __name__ == "__main__":
    unittest.main()
