from typing import Any, TypeAlias, TypeVar, Protocol, Callable, Sequence
from math import ceil

from numpy.polynomial.legendre import leggauss

from chex import ArrayTree

from .prelude import *
from .r_fun import ADF
from .elp import (
    domain_masks,
    integrate_legendre,
    compute_elp_weights,
    BrokenCellMask,
    PaddingMask
)


T = TypeVar("T", bound=Callable[..., ArrayTree])
Scalar: TypeAlias = Array
Origin: TypeAlias = Array


class Integrand(Protocol[T]):
    def __call__(self, *args: Any, **kwds: Any) -> T:
        ...


Weights: TypeAlias = Array
Nodes: TypeAlias = Array
Domain: TypeAlias = Array
Grid1d: TypeAlias = Array
QuadRule: TypeAlias = Callable[[Grid1d], tuple[Weights, Nodes]]


def midpoint(domain: Array) -> tuple[Weights, Nodes]:
    w = domain[1:] - domain[:-1]
    nodes = (domain[1:] + domain[:-1]) / 2
    return w, nodes


def trap(domain: Array) -> tuple[Weights, Nodes]:
    a, b = zeros(len(domain)), zeros(len(domain))
    d = domain[1:] - domain[:-1]
    a = a.at[:-1].set(d)
    b = b.at[1:].set(d)
    w = (a + b) / 2
    return w, domain


def simpson(domain: Array) -> tuple[Weights, Nodes]:
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
    
    def quad(domain: Array) -> tuple[Weights, Nodes]:
        def weights_nodes(a, b):
            w = (b - a) / 2 * asarray(weights)
            n = (a + b) / 2 + nodes * (b - a) / 2
            return w, n
        return vmap(weights_nodes)(domain[:-1], domain[1:])

    return quad


def make_quad_rule(domain: Array | list[Array], method: QuadRule) -> tuple[Weights, Nodes, Domain]:
    """Creates the quadrature weights and nodes for the given domain and
    quadrature method.

    Parameters
    ----------
    domain : Array | list[Array]
    method : QuadRule

    Returns
    -------
    tuple[Weights, Nodes]
    """
    if not isinstance(domain, (list, tuple)):
        assert len(domain.shape) == 1
        if len(domain.shape) == 1:
            domain = [domain]

    W, X = zip(*(method(d) for d in domain))
    assert all((w.shape == x.shape) for w, x in zip(W, X)), "Invalid quadrature method"

    if len(W[0].shape) <= 2:
        # gauss quadrature is given in 2d format to make it easier
        # to get the quadrature nodes for each subdomain.
        # Cannot be done for other methods, since they share nodes.
        D = _meshgrid(*domain)
        W = _meshgrid(*W)
        W = jnp.prod(W, axis=-1)
        X = _meshgrid(*X)
        return W, X, D
    else:
        msg = "Invalid quadrature method provided. "
        msg += "Output must be a tuple (Weights, Nodes) of two arrays (1d or 2d) of the same shape."
        raise ValueError(msg)


def make_elp_quad_rule(
    adf: ADF,
    domain: Array | list[Array],
    polynomial_degree: int = 3,
    *args: Any,
    splits: int | Sequence[int] = 1,
    support_nodes: int | Sequence[int] = 3,
    eps: float = 1e-6,
    max_depth: int = 3,
    batch_size: None | int = None,
    **kwargs: Any
) -> tuple[Weights, Nodes, Domain, BrokenCellMask, PaddingMask]:
    """Creates an accurate quadrature rule for an arbitrary geometry, which 
    is defined via the ADF, by using Equivalent Legendre Polynomials.

    Parameters
    ----------
    adf : ADF
    domain : Array | list[Array]
    polynomial_degree : int, optional
        the maximum polynomial degree which is in theory exactly integrated, by default 3;
        note that actual precision depents on the accuracy of the spacetree algorithm.
    splits : int | Sequence[int], optional
        number of splits for the recursive spacetree algorithm for each dimension, by default 1;
        e.g. 1 corresponds to each cell being split at the center.
    support_nodes : int | Sequence[int], optional
        number of support points for each dimension which are 
        used to compute the support fraction, by default 3;
        this fraction is used on the lowest level of the tree
        to approximate the integral inside the domain.
    eps : float, optional
        threshold parameter for domain masks, by default 1e-6
    max_depth : int, optional
        maximum depth of the spacetree, by default 3

    Returns
    -------
    tuple[Weights, Nodes, Domain, BrokenCellMask, PaddingMask]
    """
    W, X, D = make_quad_rule(domain, method=gauss(polynomial_degree + 1))
    _, coefs = integrate_legendre(
        adf,
        polynomial_degree + 1,
        D,
        *args,
        splits=splits,
        max_depth=max_depth,
        support_nodes=support_nodes,
        eps=eps,
        batch_size=batch_size,
        **kwargs
    )
    W_new = compute_elp_weights(coefs, W, X, D)
    broken_cell_mask, padding_mask, _ = domain_masks(
        adf, D, *args, support_nodes=support_nodes, eps=eps, **kwargs
    )
    return W_new, X, D, broken_cell_mask, padding_mask


def truncate_elp_quad_rule(
    domain: Array | list[Array],
    polynomial_degree: int,
    broken_cell_mask: BrokenCellMask,
    padding_mask: PaddingMask,
    elp_weights: Weights,
    elp_nodes: Nodes,
) -> tuple[Weights, Nodes]:
    """
    Truncates a ELP quadrature rule. Inner cells can be
    integrated with a lower Gauss quadrature rule and padding nodes
    are removed. Weights and Nodes are flatted.
    
    Notes
    -----
    This method can not be jitted.

    Parameters
    ----------
    domain : Array | list[Array]
    polynomial_degree : int
    broken_cell_mask : BrokenCellMask
    padding_mask : PaddingMask
    elp_weights : Weights
    elp_nodes : Nodes

    Returns
    -------
    tuple[Weights, Nodes]
    """
    gauss_points = int(ceil((polynomial_degree + 1) / 2))
    W, X, D = make_quad_rule(domain, method=gauss(gauss_points))
    d = D.ndim - 1
    W = _where_with_casting(padding_mask | broken_cell_mask, 0.0, W)
    idx = W != 0
    W, X = W[idx], X[idx]
    W, X = W.reshape(-1), X.reshape(-1, d)
    elp_weights = _where_with_casting(broken_cell_mask, elp_weights, 0.0)
    elp_weights = _where_with_casting(padding_mask, 0.0, elp_weights)
    idx = elp_weights != 0
    elp_weights, elp_nodes = elp_weights[idx], elp_nodes[idx]
    elp_weights, elp_nodes = elp_weights.reshape(-1), elp_nodes.reshape(-1, d)
    weights = jnp.concatenate([W, elp_weights], axis=0)
    nodes = jnp.concatenate([X, elp_nodes], axis=0)
    return weights, nodes

    
def _where_with_casting(a, b, c):
    max_dim = max(asarray(b).ndim, asarray(c).ndim)
    d = asarray(a.ndim)
    return jnp.where(a[..., *[None for _ in range(max_dim - d)]], b, c)
    

def integrate(
    fn: Integrand[T],
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
        The quadrature rule is a function `Callable[[Array], tuple[Weights, Nodes]]` which
        should return a tuple `(wieghts, nodes)` of the method
        in 1d or 2d format, by default simpson.

    Returns
    -------
    ArrayTree
    """
    W, X, _ = make_quad_rule(domain, method)
    return integrate_quad_rule(fn, W, X, *args, **kwargs)


def integrate_quad_rule(
    fn: Integrand[T],
    weights: Weights,
    nodes: Nodes,
    *args: Any,
    axis: None | int | Sequence[int] = None,
    **kwargs: Any
) -> T:
    """Integrates the function `fn` with the given quadrature rule.

    Parameters
    ----------
    fn : Integrand[T]
    weights : Weights
    nodes : Nodes
    axis : None | int | Sequence[int], optional
        If specified, integration is only done over these axes, by default None

    Returns
    -------
    T
    """
    W, X = weights, nodes
    F = _apply_along_last_axis(lambda x: fn(x, *args, **kwargs), X)
    return tree.map(lambda y: _weigthed_product(W, y, axis), F)


def _weigthed_product(W, F, axis):
    _W = W[(...,) + (None,) * (F.ndim - W.ndim)]  # extend axis
    if axis is None:
        axis = list(range(len(W.shape)))
    return jnp.sum(_W * F, axis=axis)


def _apply_along_last_axis(f, X):
    if X.shape[-1] == 1:
        # domain is one dimensional, so we pass only scalar values to fn
        return jnp.apply_along_axis(lambda x: f(x[0]), -1, X)
    else:
        return jnp.apply_along_axis(f, -1, X)


def integrate_disk(
    fn: Integrand[T],
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
        return tree.map(lambda f: f * r, fn(p, *args, **kwargs))

    domain = [
        jnp.linspace(r_inner, r, n[0]),
        jnp.linspace(phi1, phi2, n[1]),
    ]
    return integrate(g, domain, *args, method=method, **kwargs)


def integrate_sphere(
    fn: Integrand[T],
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

    def g(t, *args, **kwargs):
        r, phi, theta = t
        x = r * sin(phi) * cos(theta)
        y = r * sin(phi) * sin(theta)
        z = r * cos(phi)
        p = stack([x, y, z]) + o
        return tree.map(lambda f: f * r ** 2 * sin(phi), fn(p, *args, **kwargs))

    domain = [
        jnp.linspace(r_inner, r, n[0]),
        jnp.linspace(phi1, phi2, n[1]),
        jnp.linspace(theta1, theta2, n[2]),
    ]
    return integrate(g, domain, *args, method=method, **kwargs)


def _meshgrid(*X):
    if len(X[0].shape) <= 1:
        return jnp.stack(jnp.meshgrid(*X), axis=-1)

    indices = [jnp.arange(x.shape[0]) for x in X]
    indices = jnp.stack(jnp.meshgrid(*indices), axis=-1)
    
    def f(i):
        M = [x[j] for x, j in zip(X, i)]
        return jnp.stack(jnp.meshgrid(*M), axis=-1)
    
    return jnp.apply_along_axis(f, -1, indices)
