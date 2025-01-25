"""
This module offers an implementation of Equivalent Legendre polynomials [1]_. 

Notes
-----
.. [1] Abedian, Alireza, and Alexander Düster. 
   "Equivalent Legendre polynomials: Numerical integration of discontinuous functions in the finite element methods." 
   Computer Methods in Applied Mechanics and Engineering 343 (2019): 690-720.
"""
from typing import Any, TypeAlias, Sequence
import math

from .prelude import *
from .utils import apply_along_last_dims
from .r_fun import ADF, scale_without_normalization, translate


BrokenCellMask: TypeAlias = Array
PaddingMask: TypeAlias = Array
SupportFraction: TypeAlias = Array
Domain: TypeAlias = Array
Weights: TypeAlias = Array
Nodes: TypeAlias = Array
Moments: TypeAlias = Array
LegendreCoefs: TypeAlias = Array


@partial(jit, static_argnames="n")
def legendre_polynomial(x: Array, n: int) -> Array:
    """Returns the first `n` Legendre polynomials evaluated at `x`.

    Parameters
    ----------
    x : Array
    n : int

    Returns
    -------
    Array
    """
    if n <= 0:
        raise ValueError("Degree must be greater than 0.")
    
    p = [jnp.ones_like(x), x]
    for deg in range(1, n):
        pn = ((2 * deg + 1) * x * p[deg] - deg * p[deg - 1]) / (deg + 1)
        p.append(pn)

    pn = asarray(p)[:n]
    return jnp.moveaxis(pn, 0, -1)


@partial(jit, static_argnames="n")
def legendre_poly_antiderivative(x: Array, n: int) -> Array:
    """Returns the first `n` antiderivatives of the Legendre polynomials evaluated at `x`.

    Parameters
    ----------
    x : Array
    n : int

    Returns
    -------
    Array
    """
    if n > 1:
        p2, p1 = legendre_polynomial(x, n), legendre_polynomial(x, n - 1)
        p = p2.at[..., 1:].set(x[..., None] * p2[..., 1:] - p1)
    else:
        p = legendre_polynomial(x, n)
    p = p.at[..., 0].set(x * p[..., 0])
    return p / jnp.arange(1, n + 1)


def domain_masks(
    adf: ADF,
    domain: Domain,
    *args: Any,
    support_nodes: int | Sequence[int] = 3,
    eps=1e-6,
    **kwargs: Any
) -> tuple[BrokenCellMask, PaddingMask, SupportFraction]:
    """For a given domain grid and a domain which is implicitly
    defined via the ADF, this function computes which cells are
    between the inside and outside of the domain (broken cells)
    and which cells are completely outside the domain (padding cells).
    Additionally a support fraction is computed, which
    specifies the approximate support within the cell
    (0 = padding cell, 1 = inside the comain).

    Parameters
    ----------
    adf : ADF
    domain : Domain
        domain grid
    support_nodes : int | Sequence[int], optional
        number of support points for each dimension which are 
        used to compute the support fraction, by default 3
    eps : _type_, optional
        threshold parameter for nodes inside and outside the boundary,
        e.g. cell with only one node inside the domain but `adf < eps` is 
        still a padding cell, by default 1e-6

    Returns
    -------
    tuple[BrokenCellMask, PaddingMask, VolFraction]
    """
    d = len(domain.shape) - 1  # dimension of the problem
    LD = jnp.apply_along_axis(adf, -1, domain, *args, **kwargs)
    nodes = _make_support_nodes(domain, support_nodes)
    o = math.prod(nodes.shape[d:-1])
    LX = jnp.apply_along_axis(adf, -1, nodes, *args, **kwargs)

    def _make_mask(LD, LX):
        max_count = o + 2 ** d
        nG, nX = jnp.sum(LD >= 0 - eps), jnp.sum(LX >= 0 - eps)
        n = nG + nX
        padding_cell = (n == 0) | ((nX == 0) & jnp.all(LD <= 0 + eps))  # second case occures if adf>=0 on the bounday of the cell, but negativ inside
        broken_cell = (n != max_count) & (n > 0)
        broken_cell = jnp.where(padding_cell, False, broken_cell)
        r = lax.cond(padding_cell,
                     lambda: 0.,
                     lambda: lax.cond(broken_cell,
                                      lambda: n / max_count,
                                      lambda: 1.0))
        return broken_cell, padding_cell, r

    def _make_mask_for_indices(i):
        LDi, LXi = lax.dynamic_slice(LD, i, (2,) * d), LX[*i]
        return _make_mask(LDi, LXi)

    broken_cell_mask, padding_cell_mask, count_fraction = _apply_on_indices(
        _make_mask_for_indices, tuple(dim - 1 for dim in domain.shape[:-1])
    )
    return broken_cell_mask, padding_cell_mask, count_fraction
    

def _make_support_nodes(domain: Domain, support_nodes: int | Sequence[int]) -> Array:
    d = len(domain.shape) - 1
    
    if not isinstance(support_nodes, Sequence):
        support_nodes = [support_nodes] * d
        
    dl = domain[*[slice(0, -1) for _ in range(d)]]
    du = domain[*[slice(1, None) for _ in range(d)]]
    
    def make_points(dl, du):
        points = [jnp.linspace(lower, upper, n + 2)[1:-1] for lower, upper, n in zip(dl, du, support_nodes)]
        return jnp.stack(jnp.meshgrid(*points), axis=-1)
    
    points = asarray(apply_along_last_dims(make_points, dl, du))
    return points


def _apply_on_indices(fn, shape):
    indices = jnp.indices(shape)
    indices = jnp.moveaxis(indices, 0, -1)
    return apply_along_last_dims(fn, indices)


def integrate_legendre(
    adf: ADF,
    degree: int | Sequence[int],
    domain: Domain,
    *args: Any,
    splits: int | Sequence[int] = 1,
    support_nodes: int | Sequence[int] = 3,
    eps: float = 1e-6,
    max_depth: int = 3,
    **kwargs: Any
) -> tuple[Moments, LegendreCoefs]:
    """Computes moments and coefficients for the given domain for the
    equivalent legendre polynomials. The integration is performed with
    an adapted version of the recursive spacetrees algorithm from [1]_.
    The moments and coefficients are evaluated for each cell in the 
    domain grid.

    Parameters
    ----------
    adf : ADF
        describes the inside of the domain which is integrated
    degree : int | Sequence[int]
        the number of Legendre polynomials for each dimension, the polynomial degree is `degree + 1`
    domain : Domain
        domain cell grid
    splits : int | Sequence[int], optional
        number of splits for the recursive algorithm for each dimension, by default 1;
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
    tuple[Moments, LegendreCoefs]
    """
    def integrate_cell(idx):
        d = idx.shape[0]
        cell_domain = lax.dynamic_slice(domain, jnp.concatenate([idx, array([0])]), (2,) * d + (d,))
        _adf, centered_domain = _center(adf, cell_domain)
        moments = _integrate_legendre_cell(_adf, degree, centered_domain, *args,
                                           splits=splits, support_nodes=support_nodes,
                                           eps=eps, depth=0, max_depth=max_depth, **kwargs)
        
        # compute coefs:
        c = [(2 * jnp.arange(d) + 1) / 2 for d in moments.shape]
        c = jnp.prod(jnp.stack(jnp.meshgrid(*c), axis=-1), axis=-1)
        coefs = c * moments
        return moments, coefs
    
    return _apply_on_indices(integrate_cell, tuple(d - 1 for d in domain.shape[:-1]))


def compute_elp_weights(
    coefs: LegendreCoefs,
    weights: Weights,
    nodes: Nodes,
    domain: Domain,
) -> Weights:
    """Evaluates the Equivalent Legendre Polynomials for the given quadrature
    rule. Note that the domain must be the same which was used to evaluate the 
    ELP coefficients.

    Parameters
    ----------
    coefs : LegendreCoefs
    weights : Weights
        quadrature weights
    nodes : Nodes
        quadrature nodes
    domain : Domain
        domain grid

    Returns
    -------
    Weights
    """
    d = domain.ndim - 1
    lb = domain[*[slice(0, -1) for _ in range(d)]]
    ub = domain[*[slice(1, None) for _ in range(d)]]
    
    def compute_on_node(coefs, weight, node):
        leg_poly = [legendre_polynomial(x, p) for x, p in zip(node, coefs.shape)]
        elp = jnp.sum(coefs * jnp.prod(jnp.stack(jnp.meshgrid(*leg_poly), axis=-1), axis=-1))
        return elp * weight
    
    def compute_new_weights(coefs, weights, nodes, lb, ub):
        nodes = (nodes * 2 - (lb + ub)) / (ub - lb)
        new_weights = apply_along_last_dims(lambda w, x: compute_on_node(coefs, w, x), weights, nodes)
        return new_weights

    new_weights = apply_along_last_dims(compute_new_weights, coefs, weights, nodes, lb, ub, dims=d + 1)
    return new_weights


def _center(adf, d):
    lb, ub = d[*[0 for _ in d.shape[:-1]]], d[*[-1 for _ in d.shape[:-1]]]
    d = (d * 2 - (lb + ub)) / (ub - lb)
    _adf = scale_without_normalization(adf, 2)
    _adf = translate(_adf, -(lb + ub))
    _adf = scale_without_normalization(_adf, 1 / (ub - lb))
    return _adf, d


def _integrate_legendre_cell(
    adf: ADF,
    degree: int | Sequence[int],
    cell_domain: Domain,
    *args: Any,
    splits: int | Sequence[int] = 1,
    support_nodes: int | Sequence[int] = 3,
    eps: float = 1e-6,
    depth: int = 0,
    max_depth: int = 3,
    **kwargs: Any
) -> Array:
    d = cell_domain.ndim - 1
    if not isinstance(splits, Sequence):
        splits = [splits] * d
    broken_cell_mask, padding_cell_mask, ratio = domain_masks(
        adf, cell_domain, *args, eps=eps, support_nodes=support_nodes, **kwargs)
    
    @partial(jit, inline=False)
    def _recursive_call(idx):
        padding = padding_cell_mask[*idx]
        broken = broken_cell_mask[*idx]
        r = ratio[*idx]
        Dl = cell_domain[*idx]
        Du = cell_domain[*[i + 1 for i in idx]]
        Pl = legendre_poly_antiderivative(Dl, degree)
        Pu = legendre_poly_antiderivative(Du, degree)
        P = Pu - Pl
        IP = jnp.prod(jnp.stack(jnp.meshgrid(*P), axis=-1), axis=-1)
        IP = lax.cond(
            padding, lambda: zeros_like(IP), lambda: IP
        )
        
        if depth == max_depth:
            return IP * r
        else:
            new_domain = [jnp.linspace(lower, upper, s + 2) for lower, upper, s in zip(Dl, Du, splits)]
            new_domain = jnp.stack(jnp.meshgrid(*new_domain), axis=-1)
            return lax.cond(
                broken,
                lambda: _integrate_legendre_cell(
                    adf, degree, new_domain, *args, 
                    support_nodes=support_nodes, eps=eps, depth=depth + 1,
                    max_depth=max_depth, **kwargs),
                lambda: IP
            )
    return jnp.sum(_apply_on_indices(_recursive_call, broken_cell_mask.shape), axis=list(range(d)))
    