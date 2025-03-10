from typing import Any, TypeAlias, TypeVar, Protocol, Callable
from numpy.polynomial.legendre import leggauss

from magpi.prelude import *
from chex import ArrayTree

T = TypeVar("T", bound=Callable[..., ArrayTree])
Scalar: TypeAlias = Array
Origin: TypeAlias = Array


class Integrand(Protocol[T]):
    def __call__(self, *args: Any, **kwds: Any) -> T:
        ...


QuadRule: TypeAlias = Callable[[Array], tuple[Array, Array]]


def midpoint(domain: Array) -> tuple[Array, Array]:
    w = domain[1:] - domain[:-1]
    nodes = (domain[1:] + domain[:-1]) / 2
    return w, nodes


def trap(domain: Array) -> tuple[Array, Array]:
    a, b = zeros(len(domain)), zeros(len(domain))
    d = domain[1:] - domain[:-1]
    a = a.at[:-1].set(d)
    b = b.at[1:].set(d)
    w = (a + b) / 2
    return w, domain


def simpson(domain: Array) -> tuple[Array, Array]:
    n = len(domain) + len(domain) - 1

    def weights(a, b):
        return (b - a) / 6 * array([1, 4, 1])

    _w = vmap(weights)(domain[:-1], domain[1:])
    w = zeros(n)
    w = w.at[0].set(_w[0, 0])
    w = w.at[-1].set(_w[-1, -1])
    w = w.at[1::2].set(_w[:, 1])
    w = w.at[2:-1:2].set(_w[:-1, 2] + _w[1:, 0])
    m = (domain[:-1] + domain[1:]) / 2
    i = jnp.arange(1, len(domain))
    nodes = jnp.insert(domain, i, m)
    return w, nodes


def gauss(degree: int) -> QuadRule:
    nodes, weights = map(asarray, leggauss(degree))
    
    def quad(domain: Array) -> tuple[Array, Array]:
        def weights_nodes(a, b):
            w = (b - a) / 2 * asarray(weights)
            n = (a + b) / 2 + nodes * (b - a) / 2
            return w, n
        w, n = vmap(weights_nodes)(domain[:-1], domain[1:])
        return jnp.ravel(w), jnp.ravel(n)
    return quad


def integrate(
    f: Integrand[T],
    domain: Array | list[Array],
    *args,
    method: QuadRule = simpson,
    **kwargs
) -> T:
    """Integrates over the given domain with the provided quadrature
    rule. The domain can either be an array or a list of arrays.

    Examples
    --------
    This integrates ``f`` over the domain :math:`[0, 1]`
        >>> f = lambda x: 2 * x
        >>> d = array([0.0, 1.0])
        >>> float(integrate(f, d, method=midpoint))
        1.0

    Using more nodal points, composite rules are used
        >>> f = lambda x: sin(x)
        >>> d = linspace(0, pi, 30)
        >>> F = integrate(f, d)
        >>> bool(jnp.isclose(F, 2))
        True

    For multivariate functions, the domain can be a list indicating a
    rectangular domain. Also additional parameters can be passed using
    ``*args`` and ``*kwargs``.  This example integrates
    :math:`\\int_{-1}^{1}\\int_{0}^{1} a x_0^2 + b x_1 dx_1 dx_0`
    with :math:`a=1` and :math:`b=2`

        >>> f = lambda x, a, b: a * x[0] ** 2 + b * x[1]
        >>> d = [
        ...     linspace(-1, 1, 2),
        ...     linspace(0, 1, 2),
        ... ]
        >>> F = integrate(f, d, 1., method=gauss(2), b=2.)
        >>> bool(jnp.isclose(F, 2.66666666))
        True

    Parameters
    ----------
    f : Integrand
    domain : Array | list[Array]
        nodal points for each dimension
    method : QuadRule, optional
        The quadrature rule is a function `Callable[[Array], tuple[Array, Array]]` which
        should return a tuple `(wieghts, nodes)` of the method
        in 1d, by default simpson.

    Returns
    -------
    ArrayTree
    """
    if not isinstance(domain, (list, tuple)):
        assert len(domain.shape) > 0
        assert len(domain.shape) <= 2
        if len(domain.shape) == 1:
            domain = [domain]

    W, X = zip(*(method(d) for d in domain))
    if len(domain) > 1:
        W = stack(jnp.meshgrid(*W), axis=-1)
        X = stack(jnp.meshgrid(*X), axis=-1)
        W = jnp.prod(W, axis=-1)
    else:
        W = W[0]
        X = X[0]

    def g(x):
        return f(x, *args, **kwargs)

    F = jnp.apply_along_axis(g, -1, X)
    
    return tree.map(lambda z: jnp.tensordot(W, z, len(domain)), F)


def integrate_disk(
    f: Integrand[T],
    r: Scalar,
    o: Origin,
    n: int | tuple[int, int],
    r_inner: float = 0,
    phi1: float = 0,
    phi2: float = 2 * pi,
    *args,
    method: QuadRule = simpson,
    **kwargs
) -> T:
    """Integrates over a disk of radius ``r`` and origin ``o``.

    Examples
    --------
        >>> f = lambda x: 1.
        >>> area = integrate_disk(f, 1., array([0., 0.]), 20)
        >>> bool(jnp.isclose(area, pi))
        True

    Parameters
    ----------
    f : Integrand
    r : Scalar
        radius
    o : Origin
        origin
    n : int | tuple[int, int]
        Nodes in each dimension.
    r_inner: float
    phi1: float
    phi2: float
    method : QuadRule, optional
        Quadrature rule, by default simpson

    Returns
    -------
    ArrayTree
    """
    if not isinstance(n, tuple):
        n = (n, n)
    else:
        assert len(n) == 2

    def g(u, *args, **kwargs):
        r, phi = u
        x = r * cos(phi)
        y = r * sin(phi)
        p = stack([x, y]) + o
        return tree.map(lambda f: f * r, f(p, *args, **kwargs))

    domain = [
        jnp.linspace(r_inner, r, n[0]),
        jnp.linspace(phi1, phi2, n[1]),
    ]
    return integrate(g, domain, *args, method=method, **kwargs)


def integrate_sphere(
    f: Integrand[T],
    r: Scalar,
    o: Origin,
    n: int | tuple[int, int, int],
    r_inner: float = 0,
    phi1: float = 0,
    phi2: float = pi,
    theta1: float = 0,
    theta2: float = 2 * pi,
    *args,
    method: QuadRule = simpson,
    **kwargs
) -> T:
    """Integrates over a sphere of radius ``r`` and origin ``o``.

    Parameters
    ----------
    f : Integrand
    r : Scalar
        radius
    o : Origin
        origin
    n : int | tuple[int, int, int]
        Nodes in each dimension.
    r_inner: float
    phi1: float
    phi2: float
    theta1: float
    theta2: float
    method : QuadRule, optional
        Quadrature rule, by default simpson

    Returns
    -------
    ArrayTree
    """
    if not isinstance(n, tuple):
        n = (n, n, n)
    else:
        assert len(n) == 3

    def g(t: Array, *args, **kwargs) -> Scalar:
        r, phi, theta = t
        x = r * sin(phi) * cos(theta)
        y = r * sin(phi) * sin(theta)
        z = r * cos(phi)
        p = stack([x, y, z]) + o
        return tree.map(lambda f: f * r ** 2 * sin(phi), f(p, *args, **kwargs))

    domain = [
        jnp.linspace(r_inner, r, n[0]),
        jnp.linspace(phi1, phi2, n[1]),
        jnp.linspace(theta1, theta2, n[2]),
    ]
    return integrate(g, domain, *args, method=method, **kwargs)
