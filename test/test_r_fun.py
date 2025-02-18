from magpi.r_fun import (
    r0,
    r1,
    RP,
    cube,
    cuboid,
    sphere,
    rotate2d,
    rotate3d,
    reflect,
    translate,
    scale,
    project,
    newton_iteration
)

from . import *


class TestRFunc(JaxTestCase):
    def test_00_intersection(self):
        df1 = lambda x: 1 - x
        df2 = lambda x: x
        df = r0.intersection(df1, df2)
        self.assertIsclose(df(0.0), 0.0)
        self.assertIsclose(df(1.0), 0.0)
        self.assertTrue(df(0.5) > 0.0)
        self.assertIsclose(grad(df)(array(0.0)), 1.0)
        self.assertIsclose(grad(df)(array(1.0)), -1.0)

    def test_01_union(self):
        df1 = lambda x: x
        df2 = lambda x: x - 0.5
        df = r0.union(df1, df2)
        self.assertIsclose(df(0.0), 0.0)
        self.assertTrue(df(0.5) > 0.0)
        self.assertTrue(df(1.0) > 0.0)
        self.assertIsclose(grad(df)(array(0.0)), 1.0)
        self.assertTrue(grad(df)(array(1.0)) > 1.0)

    def test_02_xor(self):
        df1 = r0.intersection(lambda x: 1 - x, lambda x: x)
        df2 = r0.intersection(lambda x: 0.5 - x, lambda x: x + 0.5)
        df = r0.xor(df1, df2)
        self.assertIsclose(df(-0.5), 0.0)
        self.assertIsclose(df(0.0), 0.0)
        self.assertIsclose(df(0.5), 0.0)
        self.assertIsclose(df(1.0), 0.0)
        self.assertIsclose(grad(df)(-0.5), 1.0)
        self.assertIsclose(grad(df)(0.5), 1.0)
        self.assertIsclose(grad(df)(0.0), -1.0)
        self.assertIsclose(grad(df)(1.0), -1.0)

    def test_03_negate(self):
        df = r1.negate(lambda x: x)
        self.assertIsclose(df(0.0), 0.0)
        self.assertTrue(df(0.5) < 0.0)
        self.assertTrue(df(-0.5) > 0.0)

    def test_04_difference(self):
        df1 = r0.intersection(lambda x: 1 - x, lambda x: x)
        df2 = lambda x: x - 0.5
        df = r1.difference(df1, df2)
        self.assertIsclose(df(0.0), 0.0)
        self.assertIsclose(df(0.5), 0.0)
        self.assertIsclose(grad(df)(array(0.0)), 1.0)
        self.assertIsclose(grad(df)(array(0.5)), -1.0)

    def test_05_equivalence(self):
        df = r1.equivalence(lambda x: 1 - x, lambda x: x)
        self.assertIsclose(df(0.0), 0.0)
        self.assertIsclose(df(1.0), 0.0)
        self.assertTrue(df(-0.1) < 0.0)
        self.assertTrue(df(1.1) < 0.0)

    def test_05_implication(self):
        df1 = translate(sphere(r=1), 0.0)
        df2 = translate(sphere(r=1), 1.0)
        df = r0.implication(df1, df2)
        self.assertIsclose(df(0.0), 0.0)
        self.assertTrue(df(1.0) > 0.0)
        self.assertTrue(df(1.9) > 0.0)
        self.assertTrue(df(2.1) > 0.0)
        self.assertTrue(df(-1.1) > 0.0)


class TestTransformations(JaxTestCase):
    def test_00_translate(self):
        df = r1.intersection(lambda x: 1 - x, lambda x: x)
        df = translate(df, 1.0)
        self.assertIsclose(df(1.0), 0.0)
        self.assertIsclose(df(2.0), 0.0)
        self.assertTrue(df(1.5) > 0.0)
        self.assertIsclose(grad(df)(array(1.0)), 1.0)
        self.assertIsclose(grad(df)(array(2.0)), -1.0)

    def test_01_scale(self):
        df = r1.intersection(lambda x: 1 - x, lambda x: array(x))
        df = scale(df, 2)
        self.assertIsclose(df(0.0), 0.0)
        self.assertIsclose(df(2.0), 0.0)
        self.assertTrue(df(1.5) > 0.0)
        self.assertIsclose(grad(df)(array(0.0)), 1.0)
        self.assertIsclose(grad(df)(array(2.0)), -1.0)

    def test_02_scale(self):
        @jit
        def f(x):
            rp = RP(p=4)
            df = cuboid([1.0, 1.0], r_system=rp)
            df = scale(df, [1.0])
            return df(x)

        self.assertIsclose(f(array([0.5, 1])), 0.0)
        self.assertIsclose(f(array([0.5, 0])), 0.0)
        self.assertIsclose(f(array([0.0, 0.5])), 0.0)
        self.assertIsclose(f(array([1.0, 0.5])), 0.0)
        self.assertIsclose(norm(grad(f)(array([0.5, 1]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0]))), 1.0)
        self.assertIsclose(norm(grad(f)(array([0.0, 0.5]))), 1.0)
        self.assertIsclose(norm(grad(f)(array([1.0, 0.5]))), 1.0)

        # higher order normalization is preserved
        def normal_derivative(g):
            return lambda x: -grad(f)(x) @ grad(g)(x)

        ddf = normal_derivative(normal_derivative(f))
        self.assertIsclose(ddf(array([0.5, 1])), 0.0)
        self.assertIsclose(ddf(array([0.5, 0])), 0.0)
        self.assertIsclose(ddf(array([0.0, 0.5])), 0.0)
        self.assertIsclose(ddf(array([1.0, 0.5])), 0.0)

        dddf = normal_derivative(ddf)
        self.assertIsclose(dddf(array([0.5, 1])), 0.0)
        self.assertIsclose(dddf(array([0.5, 0])), 0.0)
        self.assertIsclose(dddf(array([0.0, 0.5])), 0.0)
        self.assertIsclose(dddf(array([1.0, 0.5])), 0.0)

    def test_03_rotate2d(self):
        f = cube(1.0)
        f = rotate2d(f, pi / 2, (0.5, 0.5))

        self.assertIsclose(f(array([0.5, 1])), 0.0)
        self.assertIsclose(f(array([0.5, 0])), 0.0)
        self.assertIsclose(f(array([0.0, 0.5])), 0.0)
        self.assertIsclose(f(array([1.0, 0.5])), 0.0)
        self.assertIsclose(norm(grad(f)(array([0.5, 1]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0]))), 1.0)
        self.assertIsclose(norm(grad(f)(array([0.0, 0.5]))), 1.0)
        self.assertIsclose(norm(grad(f)(array([1.0, 0.5]))), 1.0)

    def test_04_rotate3d(self):
        f = cube(1.0)
        f = rotate3d(f, (0.0, 0.0, pi / 2), o=(0.5, 0.5, 0.5))

        self.assertIsclose(f(array([1.0, 0.5, 0.5])), 0.0)
        self.assertIsclose(f(array([0.5, 1, 0.5])), 0.0)
        self.assertIsclose(f(array([0.5, 0.5, 1.0])), 0.0)
        self.assertIsclose(f(array([0.0, 0.5, 0.5])), 0.0)
        self.assertIsclose(f(array([0.5, 0, 0.5])), 0.0)
        self.assertIsclose(f(array([0.5, 0.5, 0.0])), 0.0)

        self.assertIsclose(norm(grad(f)(array([1.0, 0.5, 0.5]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 1, 0.5]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0.5, 1.0]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.0, 0.5, 0.5]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0, 0.5]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0.5, 0.0]))), 1)

    def test_05_reflect2d(self):
        f = cube(1.0)
        f = translate(f, [-1.0, 0.0])
        f = reflect(f, (1.0, 0.0))

        self.assertIsclose(f(array([0.5, 1])), 0.0)
        self.assertIsclose(f(array([0.5, 0])), 0.0)
        self.assertIsclose(f(array([0.0, 0.5])), 0.0)
        self.assertIsclose(f(array([1.0, 0.5])), 0.0)
        self.assertIsclose(norm(grad(f)(array([0.5, 1]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0]))), 1.0)
        self.assertIsclose(norm(grad(f)(array([0.0, 0.5]))), 1.0)
        self.assertIsclose(norm(grad(f)(array([1.0, 0.5]))), 1.0)

    def test_06_reflect_with_origin(self):
        f = cube(1.0)
        f = translate(f, [1, 1])
        f = reflect(f, (1.0, 1.0), o=(1.0, 1.0))

        self.assertIsclose(f(array([0.5, 1])), 0.0)
        self.assertIsclose(f(array([0.5, 0])), 0.0)
        self.assertIsclose(f(array([0.0, 0.5])), 0.0)
        self.assertIsclose(f(array([1.0, 0.5])), 0.0)
        self.assertIsclose(norm(grad(f)(array([0.5, 1]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0]))), 1.0)
        self.assertIsclose(norm(grad(f)(array([0.0, 0.5]))), 1.0)
        self.assertIsclose(norm(grad(f)(array([1.0, 0.5]))), 1.0)

    def test_07_rotate_raises_value_error(self):
        f = cube(1.0)
        with self.assertRaises(ValueError):
            rotate3d(f, pi)

    def test_08_rotate_raises_value_error(self):
        f = cube(1.0)
        with self.assertRaises(ValueError):
            rotate3d(f, (pi, pi, pi), (1.0, 1.0, 1.0))

    def test_09_project(self):
        f = cube(1.0)
        f2 = project(f, (1.0, 0), o=(0.5, 0.0))

        self.assertIsclose(f2(array([1, 1])), f2(array([0.5, 1])))
        self.assertIsclose(f2(array([2, 0.4])), f2(array([0.5, 0.4])))
        self.assertIsclose(f2(array([0, 0.2])), f2(array([0.5, 0.2])))
        self.assertIsclose(f2(array([10, 10])), f2(array([0.5, 10])))


class TestCube(JaxTestCase):
    def test_00_cube1d(self):
        df = cube(1.0, centering=True)
        self.assertIsclose(df(array(-0.5)), 0.0)
        self.assertIsclose(df(array(0.5)), 0.0)

    def test_01_t_shape(self):
        df1 = cuboid((1, 3.5))
        df1 = translate(df1, [1.0, 0.0])
        df2 = cuboid((3, 1))
        df2 = translate(df2, [0, 3])
        df = r0.union(df1, df2)
        self.assertIsclose(df(array([1.5, 4])), 0.0)
        self.assertIsclose(df(array([1.5, 0])), 0.0)
        self.assertIsclose(norm(grad(df)(array([1.5, 4]))), 1.0)
        self.assertIsclose(norm(grad(df)(array([1.5, 0]))), 1.0)
        self.assertGreater(df(array([1.5, 3])), 0.0)


class TestSphere(JaxTestCase):
    def test_00_sphere(self):
        s = sphere(2.0)
        y = s(array([2.0, 0.0]))
        self.assertIsclose(y, 0.0)
        self.assertIsclose(norm(grad(s)(array([2.0, 0.0]))), 1.0)
        self.assertIsclose(norm(grad(s)(array([0.0, 2.0]))), 1.0)

    def test_01_sphere_normalized_1nd_order(self):
        s = sphere(1.0)

        def normal_derivative(g):
            return lambda x: grad(s)(x) * grad(g)(x)

        self.assertIsclose(normal_derivative(s)(1.0), 1.0)
        self.assertIsclose(normal_derivative(s)(-1.0), 1.0)


class TestNewtonIteration(JaxTestCase):
    def test_001_newton_iteration_sphere(self):
        adf = sphere(0.5)
        x = newton_iteration(
            adf,
            array([0.2, 0.2, 0.3]),  # x0
        )
        self.assertIsclose(adf(x), 0.0)
        
    def test_002_newton_iteration_sphere_with_bounds(self):
        adf = sphere(0.5)
        x = newton_iteration(
            adf,
            array([0.15, 0.12, 0.17]),  # x0
            array([0, 0, 0.]),  # lb
            array([0.2, 0.2, 0.2])  # ub
        )
        self.assertIsclose(x, array([0.2, 0.2, 0.2]))
