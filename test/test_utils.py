from magpi.utils import apply_along_last_dims

from . import *


class TestApplyAlongLastAxis(JaxTestCase):
    def test_000_apply_along_last_dims_1d(self):
        f = lambda x: jnp.sum(x)
        a = jnp.array([1, 2, 3])
        result = apply_along_last_dims(f, a)
        self.assertEqual(result, 6)

    def test_001_apply_along_last_dims_2d(self):
        f = lambda x: jnp.sum(x)
        a = jnp.array([[1, 2], [3, 2]])
        result = apply_along_last_dims(f, a)
        self.assertPytreeEqual(result, asarray([3, 5]))

    def test_002_apply_along_last_dims_3d(self):
        def f(x):
            self.assertEqual(x.shape, (2,))
            return jnp.sum(x)

        a = jnp.array([[[1, 2], [3, 2]], [[1, 2], [3, 2]]])
        result = apply_along_last_dims(f, a)
        self.assertPytreeEqual(result, asarray([[3, 5], [3, 5]]))

    def test_003_apply_with_pytree(self):
        def f(a, b):
            return jnp.where(a, jnp.sum(b), 0)

        a = jnp.array([True, False])
        b = jnp.array([[1, 10], [3, 2]])
        result = apply_along_last_dims(f, a, b)
        self.assertPytreeEqual(result, asarray([11, 0]))

    def test_004_apply_with_pytree_3d(self):
        def f(a, b):
            return jnp.where(a, jnp.sum(b), 0)

        a = jnp.array([1, 2])
        a = jnp.stack(jnp.meshgrid(a, a), axis=-1)
        b = jnp.array([[True, False], [False, True]])
        true_result = jnp.array([[2, 0], [0, 4]])
        result = apply_along_last_dims(f, b, a)
        self.assertPytreeEqual(result, true_result)

    def test_005_apply_along_last_2_dims(self):
        def f(x):
            self.assertEqual(x.shape, (2, 2))
            return jnp.sum(x)

        a = jnp.array([[[1, 2], [3, 2]], [[1, 2], [5, 2]]])
        result = apply_along_last_dims(f, a, dims=2)
        self.assertPytreeEqual(result, asarray([8, 10]))

    def test_006_not_enough_dims_error(self):
        def f(a, b):
            return jnp.where(a, jnp.sum(b), 0)

        a = jnp.array([1, 2])
        a = jnp.stack(jnp.meshgrid(a, a), axis=-1)
        b = jnp.array([True, False])
        with self.assertRaises(ValueError):
            apply_along_last_dims(f, b, a)
    
    def test_007_apply_with_identity_on_last_2_dims(self):
        def f(x):
            assert x.shape == (2, 2)
            return x

        a = jnp.array([1, 2])
        a = jnp.stack(jnp.meshgrid(a, a), axis=-1)
        result = apply_along_last_dims(f, a, dims=2)
        self.assertPytreeEqual(result, a)
        
    def test_008_apply_over_full_array(self):
        def f(x):
            assert x.shape == ()
            return sin(jnp.sum(x))

        a = jnp.array([1, 2])
        a = jnp.stack(jnp.meshgrid(a, a), axis=-1)
        result = apply_along_last_dims(f, a, dims=0)
        self.assertPytreeEqual(result, sin(a))
