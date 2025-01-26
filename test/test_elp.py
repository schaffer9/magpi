from magpi.integrate import (
    integrate, gauss, midpoint, integrate_disk, integrate_sphere,
    integrate_quad_rule, make_quad_rule
)
from magpi.elp import (
    legendre_polynomial,
    legendre_poly_antiderivative,
    domain_masks,
    _integrate_legendre_cell,
    integrate_legendre,
    _center,
    compute_elp_weights
)
from magpi.utils import apply_along_last_dims
from magpi.r_fun import sphere, cube, translate

from . import *


class TestLegendrePolynomial(JaxTestCase):
    def test_001_legendre_polynomial0(self):
        x = zeros((5,))
        pn = legendre_polynomial(x, 1)
        self.assertPytreeEqual(pn, ones_like(x)[:, None])
        
    def test_002_legendre_polynomial5(self):
        x = zeros((1, 2, 3, 4))
        result = array([1, 0, -0.5, 0, 0.375, 0])
        result = apply_along_last_dims(lambda x: result, x, dims=0)
        pn = legendre_polynomial(x, 6)
        self.assertPytreeEqual(pn, result)
        
    def test_003_degree_smaller_than_1_raises_error(self):
        with self.assertRaises(ValueError):
            legendre_polynomial(0, 0)
    
            
class TestLegendreAntiderivative(JaxTestCase):
    def test_000_antiderivative(self):
        Pn = jacfwd(legendre_poly_antiderivative)(0.0, 4)
        self.assertEqual(Pn.shape, (4,))
        pn = legendre_polynomial(0.0, 4)
        self.assertEqual(pn.shape, (4,))
        self.assertPytreeEqual(Pn, pn)
        
    def test_001_antiderivative_on_array_input(self):
        x = zeros((3, 3, 3))
        Pn = legendre_poly_antiderivative(x, 4)
        self.assertEqual(Pn.shape, (3, 3, 3, 4))
        Pn_true = legendre_poly_antiderivative(0.0, 4)
        result = apply_along_last_dims(lambda x: Pn_true, x, dims=0)
        self.assertPytreeEqual(Pn, result)
        
    def test_002_antiderivative_of_first_leg_poly(self):
        x = zeros((3,))
        Pn = legendre_poly_antiderivative(x, 1)
        self.assertEqual(Pn.shape, (3, 1))
        self.assertPytreeEqual(Pn, zeros((3, 1)))
        Pn_1 = legendre_poly_antiderivative(-1, 1)
        Pn1 = legendre_poly_antiderivative(1, 1)
        self.assertEqual(Pn1 - Pn_1, 2)

    def test_003_antiderivative_deg5(self):
        x = zeros((2,))
        Pn = legendre_poly_antiderivative(x, 5)
        Pn_true = legendre_poly_antiderivative(0.0, 5)
        self.assertPytreeEqual(Pn, asarray([Pn_true, Pn_true]))
        

class TestMakeDomainMasks(JaxTestCase):
    def test_000_make_domain_masks(self):
        domain = array([-1, 1])
        _, X, D = make_quad_rule(domain, method=gauss(1))
        adf = cube(2, centering=True)
        broken_cell_mask, padding_mask, r = domain_masks(adf, D, support_nodes=1)
        result = (array([False]), array([False]))
        self.assertPytreeEqual((broken_cell_mask, padding_mask), result)
        self.assertPytreeEqual(r, array([1]))
        adf = lambda x: -cube(2, centering=True)(x)
        broken_cell_mask, padding_mask, r = domain_masks(adf, D, support_nodes=[1, 1])
        result = (array([False]), array([True]))
        self.assertPytreeEqual((broken_cell_mask, padding_mask), result)
        self.assertPytreeEqual(r, array([0]))
         
    def test_001_make_domain_masks_for_disk(self):
        domain = [jnp.linspace(-1, 1, 5)] * 2
        radius = sqrt(2 * 0.5 ** 2)
        adf = sphere(radius)
        _, X, D = make_quad_rule(domain, method=gauss(3))
        broken_cell_mask, padding_mask, r = domain_masks(adf, D, support_nodes=3)
        
        true_broken_cell_mask = array([
            [False, True, True, False],
            [True, False, False, True],
            [True, False, False, True],
            [False, True, True, False],
        ])
        true_padding_mask = array([
            [True, False, False, True],
            [False, False, False, False],
            [False, False, False, False],
            [True, False, False, True],
        ])
        result = (true_broken_cell_mask, true_padding_mask)
        
        def adf2(x):
            y = -sphere(radius)(x)
            return y

        broken_cell_mask, padding_mask, r = domain_masks(adf2, D, support_nodes=[3, 3])
        true_padding_mask = array([
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ])
        result = (true_broken_cell_mask, true_padding_mask)
        self.assertPytreeEqual((broken_cell_mask, padding_mask), result)


class TestIntegrateLegendreCell(JaxTestCase):
    def test_000_integrate1d(self):
        adf = lambda x: 1.0
        domain = jnp.linspace(-1, 0, 2)
        W, X, D = make_quad_rule([domain], method=gauss(3))
        true_result = integrate_quad_rule(lambda x: legendre_polynomial(x, 5), W, X)
         
        result = _integrate_legendre_cell(
            adf, 5, D, splits=1, support_nodes=1, max_depth=0
        )
        self.assertIsclose(result, true_result)
        
    def test_001_integrate2d(self):
        adf = lambda x: 1.0
        domain = jnp.linspace(-1, 0, 2)
        W, X, D = make_quad_rule([domain, domain], method=gauss(3))
        
        def integrand(x):
            return jnp.outer(legendre_polynomial(x[0], 5), legendre_polynomial(x[1], 5))
        
        true_result = integrate_quad_rule(integrand, W, X)
         
        result = _integrate_legendre_cell(
            adf, 5, D, splits=1, support_nodes=1, max_depth=0
        )
        self.assertIsclose(result, true_result)
        
    def test_002_integrate1d_max_depth1(self):
        def adf(x):
            return jnp.sum(x)
        domain = jnp.linspace(-1, 1, 2)
        _, _, D = make_quad_rule([domain], method=gauss(3))
        
        def integrand(x):
            return legendre_polynomial(x, 5)
        
        W, X, _ = make_quad_rule([jnp.linspace(0, 1, 2)], method=gauss(3))
        true_result = integrate_quad_rule(integrand, W, X)
         
        result = _integrate_legendre_cell(
            adf, 5, D, splits=1, support_nodes=1, max_depth=1
        )
        print(result, true_result)
        self.assertIsclose(result, true_result)
        
    def test_003_integrate1d_max_depth1_splits2(self):
        def adf(x):
            return jnp.sum(x)
        domain = jnp.linspace(-1, 1, 2)
        _, _, D = make_quad_rule([domain], method=gauss(3))
        
        def integrand(x):
            return legendre_polynomial(x, 5)
        
        W, X, _ = make_quad_rule([jnp.linspace(0, 1, 2)], method=gauss(3))
        true_result = integrate_quad_rule(integrand, W, X)
         
        result = _integrate_legendre_cell(
            adf, 5, D, splits=[2, 2], support_nodes=1, max_depth=6
        )
        self.assertIsclose(result, true_result, atol=1e-3)


class TestIntegrateLegendre(JaxTestCase):
    def test_000_integrate1d(self):
        adf = lambda x: 1.0
        domain = jnp.linspace(-1, 1, 3)
        W, X, D = make_quad_rule([domain], method=gauss(3))
        true_result = integrate_quad_rule(
            lambda x: legendre_polynomial(x, 5) / 2, W, X) * 2
         
        result, coefs = integrate_legendre(
            adf, 5, D, splits=1, support_nodes=1, max_depth=0
        )
        
        self.assertIsclose(result, array([true_result, true_result]))
        self.assertIsclose(coefs, array([[1., 0., 0., 0., 0.],
                                         [1., 0., 0., 0., 0.]]))

class TestCenter(JaxTestCase):
    def test_000_center(self):
        domain = jnp.linspace(1, 2, 2)[:, None]
        adf = lambda x: jnp.sum(x)
        _adf, domain_center = _center(adf, domain)
        self.assertPytreeEqual(domain_center, array([-1.0, 1.0])[:, None])
        self.assertEqual(_adf(-1), adf(1))
        self.assertEqual(_adf(1), adf(2))
    
    def test_001_center_2d(self):
        domain = jnp.linspace(1, 2, 2)
        _, _, D = make_quad_rule([domain, domain], method=gauss(1))
        adf = translate(cube(1), array([1, 1]))
        _adf, domain_center = _center(adf, D)
        self.assertPytreeEqual(domain_center, (2 * D - 3))
        self.assertEqual(_adf(array([-1, -1])), 0)
        self.assertEqual(_adf(array([1, -1])), 0)
        self.assertEqual(_adf(array([-1, 1])), 0)
        self.assertEqual(_adf(array([1, 1])), 0)
        
        
class TestIntegrateWithELP(JaxTestCase):
    def test_000_integrate1d(self):
        def f(x):
            return x + x ** 2 + x ** 3 + x ** 4 + x ** 5
        
        true_result = integrate(f, [jnp.linspace(0, 0.5, 2)], method=gauss(3))
        
        adf = cube(0.5)
        domain = jnp.linspace(-1, 1, 4)
        W, X, D = make_quad_rule([domain], gauss(6))
        _, coefs = integrate_legendre(adf, 6, D, splits=1, max_depth=2)
        W_new = compute_elp_weights(coefs, W, X, D)
        result = integrate_quad_rule(f, W_new, X)
        self.assertPytreeEqual(result, true_result)
        
    def test_001_integrate_over_square(self):
        def f(x):
            return jnp.sum(x + x ** 2 + x ** 3 + x ** 4 + x ** 5)
        
        true_result = integrate(f, [jnp.linspace(0, 0.5, 2), jnp.linspace(0, 0.5, 2)], method=gauss(3))
        
        adf = cube(0.5)
        domain = jnp.linspace(-1, 1, 4)
        W, X, D = make_quad_rule([domain, domain], gauss(6))
        _, coefs = integrate_legendre(adf, 6, D, splits=1, max_depth=2)
        W_new = compute_elp_weights(coefs, W, X, D)
        result = integrate_quad_rule(f, W_new, X)
        self.assertIsclose(result, true_result)
    
    def test_002_integrate_3dshpere(self):
        def f(x):
            return jnp.sum(jnp.sin(x))
        
        true_result = integrate_sphere(f, 1.0, array([0, 0, 0.]), 10, method=gauss(3))
        
        adf = sphere(1)
        domain = jnp.linspace(-1, 1, 4)
        W, X, D = make_quad_rule([domain, domain, domain], gauss(6))
        _, coefs = integrate_legendre(adf, 6, D, splits=1, max_depth=4)
        W_new = compute_elp_weights(coefs, W, X, D)
        result = integrate_quad_rule(f, W_new, X)
        self.assertIsclose(result, true_result)
        