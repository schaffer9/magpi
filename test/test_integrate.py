from magpi.integrate import (
    integrate, gauss, midpoint, integrate_disk, integrate_sphere,
    integrate_quad_rule, make_quad_rule
)

from . import *


class TestIntegrate(JaxTestCase):
    def test_000_integrate_sin(self):
        f = lambda x: sin(x)
        d = linspace(0, pi, 30)
        F = integrate(f, d)
        self.assertEqual(F.shape, ())
        self.assertTrue(jnp.isclose(F, 2))
        
    def test_001_multioutput(self):
        f = lambda x: array(
            [[sin(x[0]), sin(x[1])], 
             [cos(x[1]), cos(x[0])]]
        )
        d = [
            linspace(0, 2 * pi, 50),
            linspace(0, 2 * pi, 50),
        ]
        F = integrate(f, d, method=gauss(3))
        self.assertTrue(jnp.all(jnp.isclose(F, zeros((2, 2)), atol=1e-03)))

    def test_002_integrate_scalar_fun_with_midpoint(self):
        f = lambda x: jnp.sum(2 * x)
        d = array([0.0, 1.0])
        F = integrate(f, d, method=midpoint)
        self.assertEqual(F.shape, ())
        self.assertIsclose(F, 1.0)

    def test_003_integrate_with_gauss5(self):
        def f(x):
            y = x[0] + x[1]
            return asarray([y, y])
        d = array([0.0, 1.0])
        F = integrate(f, [d, d], method=gauss(5))
        self.assertEqual(F.shape, (2,))
        self.assertIsclose(F, asarray([1, 1]))

    def test_004_integrate_disk(self):
        f = lambda x: 1
        F = integrate_disk(f, 1., array([0, 0.]), 5, method=gauss(5))
        self.assertEqual(F.shape, ())
        self.assertIsclose(F, pi)

    def test_004_integrate_sphere(self):
        f = lambda x: 1
        F = integrate_sphere(f, 1., array([0, 0., 0]), 5, method=gauss(5))
        self.assertEqual(F.shape, ())
        self.assertIsclose(F, 4 / 3 * pi)

    def test_005_integrate_pytree(self):
        def f(x):
            y = x[0] + x[1]
            return (y, (y, y))
        
        d = array([0.0, 1.0])
        F = integrate(f, [d, d], method=gauss(5))
        self.assertEqual(tree.map(lambda y: y.shape, F), ((), ((), ())))
        tree.map(lambda y: self.assertIsclose(y, 1), F)

    def test_006_integrate_quad_rule(self):
        domain = jnp.linspace(-1, 1, 5)
        W, X, _ = make_quad_rule(domain, gauss(5))
        fn = lambda x: x ** 2
        true_sol = 2 / 3
        result = integrate_quad_rule(fn, W, X)
        self.assertEqual(result, true_sol)

    def test_007_integrate_quad_rule_with_reshaping(self):
        domain = [jnp.linspace(-1, 1, 5), jnp.linspace(-1, 1, 5)]
        W, X, _ = make_quad_rule(domain, gauss(5))
        W, X = W.reshape(-1), X.reshape(-1, 2)
        fn = lambda x: jnp.sum(x ** 2)
        true_sol = 8 / 3
        result = integrate_quad_rule(fn, W, X)
        self.assertEqual(result.shape, ())
        self.assertIsclose(result, true_sol)
