from typing import Annotated, Callable, Any, NamedTuple, TypeAlias
import itertools

from .prelude import *
from .quaternions import (
    quaternion_rotation,
    from_euler_angles,
    from_axis_angle,
)


Scalar = Array | float | int
Vec = Array | list[Scalar] | Scalar | tuple[Scalar, ...]
Vec2d = Array | tuple[Scalar, Scalar]
Vec3d = Array | tuple[Scalar, Scalar, Scalar]

_annotation = """
ADF stands for Approximate distance function. It is
positive inside the respective domain, zero on the boundary
and negative outside the domain. When normalized to first order
the normal derivative has a magnitude of one everywhere on the
boundary. Higher order normalization yields a function
where higher order normal derivatives are zero.
"""
ADF = Annotated[Callable[[Array | Scalar], Scalar], _annotation]


class RFun:
    """
    Implements the basic set theoretic operations for a system of R-Functions.
    In essence, only `conjunction` and `disjunction` needs to be implemented.
    """

    def conjunction(self, a: Scalar, b: Scalar) -> Scalar:
        raise NotADirectoryError

    def disjunction(self, a: Scalar, b: Scalar) -> Scalar:
        raise NotADirectoryError

    def _conjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            return self.conjunction(a, b)

        return lambda x: op(adf1(x), adf2(x))

    def _disjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            return self.disjunction(a, b)

        return lambda x: op(adf1(x), adf2(x))

    def negate(self, adf: ADF) -> ADF:
        return lambda x: -adf(x)

    def union(self, adf1: ADF, adf2: ADF) -> ADF:
        return self._disjunction(adf1, adf2)

    def intersection(self, adf1: ADF, adf2: ADF) -> ADF:
        return self._conjunction(adf1, adf2)

    def equivalence(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            c1 = self.disjunction(a, -b)
            c2 = self.disjunction(-a, b)
            return self.conjunction(c1, c2)

        return lambda x: op(adf1(x), adf2(x))

    def implication(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            return self.disjunction(-a, b)
            
        return lambda x: op(adf1(x), adf2(x))

    def difference(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            return self.conjunction(a, -b)
            
        return lambda x: op(adf1(x), adf2(x))

    def xor(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            c1 = self.conjunction(a, -b)
            c2 = self.conjunction(-a, b)
            return self.disjunction(c1, c2)
            
        return lambda x: op(adf1(x), adf2(x))


class RAlpha(RFun):
    """
    Implements the system :math:`R_\\alpha` for some value of `alpha` in (-1, 1].

    Parameters
    ----------
    alpha : Scalar
    """

    def __init__(self, alpha: Scalar):
        assert -1.0 < alpha <= 1.0
        self.alpha = alpha

    def conjunction(self, a: Scalar, b: Scalar) -> Scalar:
        return (
            1 / (1 + self.alpha) * (a + b - sqrt(a**2 + b**2 - 2 * self.alpha * a * b))
        )

    def disjunction(self, a: Scalar, b: Scalar) -> Scalar:
        return (
            1 / (1 + self.alpha) * (a + b + sqrt(a**2 + b**2 - 2 * self.alpha * a * b))
        )


class RAlphaM(RFun):
    """
    Implements the system :math:`R_\\alpha^m` for some value of `alpha` in (-1, 1].
    Note that this system is not normalized.

    Parameters
    ----------
    m: Scalar
    alpha : Scalar
    """

    def __init__(self, m: Scalar, alpha: Scalar):
        assert -1.0 < alpha <= 1.0
        self.m = m
        self.alpha = alpha

    def conjunction(self, a: Scalar, b: Scalar) -> Scalar:
        r = a**2 + b**2
        return (
            1
            / (1 + self.alpha)
            * (a + b - sqrt(r - 2 * self.alpha * a * b))
            * r ** (self.m / 2)
        )

    def disjunction(self, a: Scalar, b: Scalar) -> Scalar:
        r = a**2 + b**2
        return (
            1
            / (1 + self.alpha)
            * (a + b + sqrt(r - 2 * self.alpha * a * b))
            * r ** (self.m / 2)
        )


class RP(RFun):
    """
    Implements the system :math:`R_p` for some even integer value `p`.
    This system is normalized to the order `p-1`.

    Parameters
    ----------
    p : int
    """

    def __init__(self, p: int):
        assert p % 2 == 0, "`p` must be an even integer"
        self.p = p

    def conjunction(self, a: Scalar, b: Scalar) -> Scalar:
        return a + b - (a**self.p + b**self.p) ** (1 / self.p)

    def disjunction(self, a: Scalar, b: Scalar) -> Scalar:
        return a + b + (a**self.p + b**self.p) ** (1 / self.p)


class RhoBlending(RFun):
    """Blending R-function which smoothes sharp corners and edges [1]_.

    Parameters
    ---------
    rho : float
        smoothing factor

    Notes
    -----
    .. [1] Shapiro, Vadim. "Semi-analytic geometry with R-functions."
       ACTA numerica 16 (2007): 239-303.
    """

    def __init__(self, rho: float):
        self.rho = rho

    def conjunction(self, a: Scalar, b: Scalar) -> Scalar:
        s = a**2 + b**2 - self.rho**2
        return a + b - sqrt(a**2 + b**2 + 1 / (8 * self.rho) * s * (s - jnp.abs(s)))

    def disjunction(self, a: Scalar, b: Scalar) -> Scalar:
        s = a**2 + b**2 - self.rho**2
        return a + b + sqrt(a**2 + b**2 + 1 / (8 * self.rho) * s * (s - jnp.abs(s)))


r1 = RAlpha(1.0)  # min, max
r0 = RAlpha(0.0)  # analytic everywhere but the origin and normalized to first order
rp2 = RP(2)  # same as r0
rp4 = RP(4)  # analytic everywhere and normalized to 3rd order.


def cuboid(edge_lengths: Vec, centering: bool = False, r_system: RFun = r0) -> ADF:
    """
    Returns the ADF of a cuboid.

    Parameters
    ----------
    edge_lengths : Vec
        geometry of the cuboid. The lenght of the vector determines the dimension.
    centering : bool, optional
        centers the cuboid at the origin, by default False
    r_system : int, optional
        the system of R-functions to construct the cuboid

    Returns
    -------
    ADF
    """
    _edge_lengths = asarray(edge_lengths)

    if centering:
        lb = -_edge_lengths / 2
        ub = _edge_lengths / 2
    else:
        lb = zeros_like(_edge_lengths)
        ub = _edge_lengths

    _intersection = compose(r_system.conjunction)

    @_intersection
    def adf(x):
        # the output is a list of R-functions which is iteratively reduced
        a = (ub - x).ravel()
        b = (x - lb).ravel()
        return jnp.stack([a, b], axis=-1).ravel()

    return adf


def cube(edge_lenght: Scalar, centering: bool = False, r_system: RFun = r0) -> ADF:
    """
    Return the ADF of a cube.

    Parameters
    ----------
    edge_lenght : Scalar
        edge length of the cube
    centering : bool, optional
        centers the cuboid at the origin, by default False
    r_system : int, optional
        the system of R-functions to construct the cube

    Returns
    -------
    ADF
    """
    return cuboid(edge_lenght, centering, r_system)


def sphere(r: Scalar) -> ADF:
    """
    Returns the ADF of a sphere which is normalized to first order.
    The dimension is arbitrary.

    Parameters
    ----------
    r : Scalar
        radius

    Returns
    -------
    ADF
    """
    return lambda x: (r**2 - norm(x) ** 2) / (2 * r)


def cylinder(r: Scalar) -> ADF:
    """
    1st order ADF of a cylinder of infinte length.
    The base of the cylinder lies in the first two dimensions of the
    input.

    Parameters
    ----------
    r : Scalar
        radius

    Returns
    -------
    ADF
    """
    s = sphere(r)
    return lambda x: s(asarray(x)[:2])


def ellipsoid(axes_lengths: Vec) -> ADF:
    """
    1st order ADF of a ellipsoid.

    Parameters
    ----------
    axes_lengths : Vec
        length of the vector determines the dimension

    Returns
    -------
    ADF
    """
    adf = sphere(1.0)
    adf = scale_without_normalization(adf, axes_lengths)
    adf = normalize_1st_order(adf)
    return adf


def compose(func: Callable[[Scalar, Scalar], Scalar]) -> Callable[..., ADF]:
    def composition(*adf):
        def _adf(x):
            d = concatenate(tree_leaves(tree_map(lambda df: df(x).ravel(), adf)))
            return reduce(func, d)

        return _adf

    return composition


def translate(adf: ADF, y: Vec) -> ADF:
    """
    Translates the ADF by the vector y.

    Parameters
    ----------
    adf : ADF
    y : Vec

    Returns
    -------
    ADF
    """
    _y = asarray(y)
    return lambda x: adf(x - _y)


def scale(adf: ADF, scaling_factor: Scalar) -> ADF:
    """
    Scales the ADF by the given `scaling_factor`.
    First order normalization is preserved.

    Parameters
    ----------
    adf : ADF
    scaling_factor : Scalar

    Returns
    -------
    ADF
    """
    _scaling_factor = asarray(scaling_factor).ravel()
    assert _scaling_factor.shape == (
        1,
    ), "`scaling_factor` must be a scalar to preserve normalization"
    _scaling_factor = _scaling_factor[0]
    return lambda x: adf(x / _scaling_factor) * _scaling_factor


def scale_without_normalization(adf: ADF, scaling_factor: Vec | Scalar) -> ADF:
    """
    Scales the ADF without normalization. This allows
    different scaling factors for each dimension but does
    not preserve normalization. First order normalization can be
    estabished by `normalize_1st_order`.

    Parameters
    ----------
    adf : ADF
    scaling_factor : Vec | Scalar

    Returns
    -------
    ADF
    """
    _scaling_factor = asarray(scaling_factor).ravel()
    return lambda x: adf(x / _scaling_factor)


def rotate2d(adf: ADF, angle: Scalar, o: Vec2d = (0.0, 0.0)) -> ADF:
    """
    Rotates the given 2d ADF by some `angle` around the point `o`.

    Parameters
    ----------
    adf : ADF
    angle : Scalar
    o : Vec2d, optional
        by default (0.0, 0.0)

    Returns
    -------
    ADF
    """
    _o = asarray(o)
    _angle = asarray(angle)
    _adf = translate(adf, -_o)
    M = array(
        [[cos(_angle), sin(_angle)], [-sin(_angle), cos(_angle)]], dtype=_angle.dtype
    )

    def rot_op(x):
        msg = f"Cannot rotate vector of size {x.shape} in 2d. Please pass a 2d vector."
        assert x.shape == (2,), msg
        return _adf(M @ x)

    return translate(rot_op, _o)


def rotate3d(
    adf: ADF,
    angle: Scalar | Vec3d,
    rot_axis: None | Vec3d = None,
    o: Vec3d = (0.0, 0.0, 0.0),
) -> ADF:
    """
    Rotates the 3d ADF by some angle around the rotation axis `rot_axis` or
    if a three euler angles are provided around the point `o`.

    Parameters
    ----------
    adf : ADF
    angle : Scalar | Vec3d
    rot_axis : None | Vec3d, optional
        by default None
    o : Vec3d, optional
        by default (0.0, 0.0, 0.0)

    Returns
    -------
    ADF

    Raises
    ------
    ValueError
    """
    _o = asarray(o)
    _adf = translate(adf, -_o)

    _angle = -asarray(angle)
    if _angle.shape == ():
        if rot_axis is None:
            msg = "If only the angle is specified, the rotation axis must be provided"
            raise ValueError(msg)
        rot_quaternion = from_axis_angle(_angle, rot_axis)
    elif _angle.shape == (3,):
        if rot_axis is not None:
            raise ValueError("If Euler angles are given, the `rot_axis` must be `None`")
        rot_quaternion = from_euler_angles(_angle)
    else:
        raise ValueError("Provide axis-angle representation or Euler angles.")

    def rot_op(x):
        msg = f"Cannot rotate vector of size {x.shape} in 3d. Please pass a 3d vector."
        assert x.shape == (3,), msg
        x = quaternion_rotation(x, rot_quaternion)
        return _adf(x)

    return translate(rot_op, _o)


def reflect(adf: ADF, normal_vec: Vec, o: Vec = 0.0) -> ADF:
    """
    Reflects the ADF along the provided normal vector of the reflection plane with origin `o`.


    Parameters
    ----------
    adf : ADF
    normal_vec : Vec
    o : Vec, optional
        by default 0.0

    Returns
    -------
    ADF
    """
    _o, _n = asarray(o), asarray(normal_vec)
    _adf = translate(adf, -_o)

    def ref_op(x):
        x = x - 2 * (x @ _n) / (norm(_n) ** 2) * _n
        return _adf(x)

    return translate(ref_op, _o)


def project(adf: ADF, normal_vec: Vec, o: Vec = 0.0) -> ADF:
    """
    Projects the ADF onto the plane defined by the normal vector `normal_vec` and `o`.

    Parameters
    ----------
    adf : ADF
    normal_vec : Vec
    o : Vec, optional
        by default 0.0

    Returns
    -------
    ADF
    """
    _o, _n = asarray(o), asarray(normal_vec)
    _adf = translate(adf, -_o)

    def proj_op(x):
        x = x - (x @ _n) / (norm(_n) ** 2) * _n
        return _adf(x)

    return translate(proj_op, o)


def revolution(adf: ADF, axis: int = 0) -> ADF:
    """Creates the body of revolution from a 2d shape.

    Parameters
    ----------
    adf : ADF
    axis : int, optional
        0 for revolution around x axis or 1 for y axis, by default 0
    """

    def rev_op(x):
        assert x.shape == (3,), "Revolution only supported in 3d."
        if axis == 0:
            r = sqrt(x[1] ** 2 + x[2] ** 2)
            return adf(jnp.stack([x[0], r]))
        elif axis == 1:
            r = sqrt(x[0] ** 2 + x[2] ** 2)
            return adf(jnp.stack([r, x[1]]))
        else:
            msg = "`axis` must either be 0 for revolution around x "
            msg += "or 1 for revolution around y."
            raise ValueError(msg)

    return rev_op


def normalize_1st_order(adf: ADF) -> ADF:
    """
    Normalizes the ADF to first order. Note that the gradient
    cannot vanish on the boundary for this function to work.

    Parameters
    ----------
    adf : ADF

    Returns
    -------
    ADF
    """
    df = jacfwd(adf)

    def normalize(x):
        y = adf(x)
        dy = df(x)
        return y / sqrt(y**2 + norm(dy) ** 2)

    return normalize


def newton_iteration(
    adf: ADF,
    x0: Vec,
    lower_bound: Vec | None = None,
    upper_bound: Vec | None = None,
    *args,
    tol: float = 1e-7,
    maxiter=10,
    **kwargs
) -> Array:
    """A box constraint newton iteration algorithm which performs the newton iteration
    along the gradient vector. This can be efficiently used to find points on the 
    boundary of the ADF.

    If no constraints are provided, the algorithm is unconstraint.
    
    Parameters
    ----------
    adf : ADF
    x0 : Vec
        Initial point
    lower_bound : Vec | None, optional
        by default None
    upper_bound : Vec | None, optional
        by default None
    tol : float, optional
        iteration is stopped if `|adf| < tol` , by default 1e-7

    Returns
    -------
    Array
        Point on the boundary with `adf=0` or on the box boundary
    """
    lb = asarray(-jnp.inf) if lower_bound is None else asarray(lower_bound)
    ub = asarray(jnp.inf) if upper_bound is None else asarray(upper_bound)
    x = asarray(x0)
    f, Jf = adf(x, *args, **kwargs), jacfwd(adf)(x, *args, **kwargs)
    
    def body(state):
        f, Jf, x, k = state
        m = Jf
        m = asarray(jnp.where(jnp.isclose(x, lb), 0.0, m))
        m = asarray(jnp.where(jnp.isclose(x, ub), 0.0, m))
        d = - (f / (Jf @ m) * m)
        t = jnp.min(jnp.asarray(jnp.where(
            jnp.isclose(d, 0), jnp.inf,
            jnp.where(d < 0, (lb - x) / d, (ub - x) / d)
        )))

        t = jnp.minimum(t, 1)
        x = x + t * d
        f = adf(x, *args, **kwargs)
        Jf = jacfwd(adf)(x, *args, **kwargs)
        return f, Jf, x, k + 1
    
    def condition(state):
        f, Jf, x, k = state
        m = Jf
        m = asarray(jnp.where(jnp.isclose(x, lb), 0.0, m))
        m = asarray(jnp.where(jnp.isclose(x, ub), 0.0, m))
        return ((jnp.abs(f) >= tol) & (~jnp.allclose(m, 0.0))) & (k < maxiter)
    
    _, _, x, _ = lax.while_loop(condition, body, (f, Jf, x, 0))
    return x


class Cell(NamedTuple):
    lower_bound: Array
    upper_bound: Array
    broken: Array
    padding: Array

    def cell_count(self):
        return len(self.broken)

    def support(self):
        return jnp.prod((self.upper_bound - self.lower_bound), axis=-1)


Cells: TypeAlias = Cell


@partial(jit, static_argnames=("adf", "max_cells"))
def partition_domain(
    adf: ADF,
    lower_bounds: Array,
    upper_bounds: Array,
    *args: Any,
    eps: float = 1e-6,
    max_depth: int = 8,
    max_cells: int = 100_000,
    **kwargs: Any,
) -> Cells:
    """Partitions the domain contained by the approximate distance function `adf`
    and `lower_bounds` and `upper_bounds` into cells with a space tree algorithm.
    The resolution becomes smaller and smaller it the boundary of the domain is approached
    until the maximum resolution is reached. The maximum resolution is based on `max_depth`
    and `max_cells`.

    The algorithm refines the domain iteratively since a recursive approach would require
    huge amounts of memory in JAX.

    Parameters
    ----------
    adf : ADF
        approximate distance function
    lower_bounds : Array
    upper_bounds : Array
    eps : float, optional
        controlls the precision whether a cell is contained by the `adf`, by default 1e-6
    max_depth : int, optional
        maximum depth of the space tree, by default 8
    max_cells : int, optional
        maximum number of cells; if less cells are required, the remaining cells are only for
        padding; by default 100_000

    Returns
    -------
    Cells
        Domain partitioned into computational cells
    """
    lb, ub = asarray(lower_bounds), asarray(upper_bounds)
    lower_bounds = zeros((max_cells, lb.shape[0]))
    upper_bounds = zeros((max_cells, ub.shape[0]))
    broken_mask = jnp.full((max_cells,), False)
    padding_mask = jnp.full((max_cells,), True)
    d = lb.shape[0]
    n = 2**d
    assert lower_bounds.shape == upper_bounds.shape

    lower_bounds = lower_bounds.at[0].set(lb)
    upper_bounds = upper_bounds.at[0].set(ub)

    is_broken_cell, is_padding_cell = _broken_or_padding_cell(adf, lb, ub, *args, eps=eps, **kwargs)

    broken_mask = broken_mask.at[0].set(is_broken_cell)
    padding_mask = padding_mask.at[0].set(is_padding_cell)
    cells = Cell(lower_bounds, upper_bounds, broken_mask, padding_mask)

    def split(cell):
        lb, ub, broken, padding = cell

        def _no_split():
            lower_bounds = zeros((n, lb.shape[0]))
            lower_bounds = lower_bounds.at[0].set(lb)
            upper_bounds = zeros((n, ub.shape[0]))
            upper_bounds = upper_bounds.at[0].set(ub)
            broken_mask = jnp.full((n,), False)
            broken_mask = broken_mask.at[0].set(broken)
            padding_mask = jnp.full((n,), True)
            padding_mask = padding_mask.at[0].set(padding)
            new_cells = Cell(lower_bounds, upper_bounds, broken_mask, padding_mask)
            return new_cells

        def _split():
            new_cells = _split_cell(adf, lb, ub, *args, eps=eps, **kwargs)
            return new_cells

        return lax.cond(broken & jnp.logical_not(padding), _split, _no_split)

    def update(i, cells):
        new_cells = vmap(split)(cells)
        new_cells = new_cells._replace(
            lower_bound=new_cells.lower_bound.reshape(-1, d),
            upper_bound=new_cells.upper_bound.reshape(-1, d),
            broken=new_cells.broken.reshape(-1),
            padding=new_cells.padding.reshape(-1),
        )

        mask = new_cells.padding
        new_cell_count = jnp.count_nonzero(jnp.logical_not(mask))

        # if the maximum depth is reached, cells with the center outside the domain
        # are removed. This increases the accuracy.
        mask = lax.cond(
            (i == (max_depth - 1)) | (new_cell_count > max_cells),
            lambda: vmap(lambda c: jnp.logical_not(_inside_adf(adf, c, *args, **kwargs)))(new_cells) | mask,
            lambda: mask,
        )
        new_cells = new_cells._replace(padding=mask)
        new_cell_count = jnp.count_nonzero(jnp.logical_not(mask))
        cell_overflow = new_cell_count > max_cells
        idx = jnp.argsort(mask)
        idx = idx[:max_cells]

        _cells = cells._replace(
            lower_bound=new_cells.lower_bound[idx],
            upper_bound=new_cells.upper_bound[idx],
            broken=new_cells.broken[idx],
            padding=new_cells.padding[idx],
        )
        return lax.cond(cell_overflow, lambda: cells, lambda: _cells)

    cells = lax.fori_loop(0, max_depth, update, cells)
    cells = vmap(_pad_cell)(cells)  # set all padding cells to zero

    return cells


def _pad_cell(cell):
    lb, ub, _, padding_cell = cell
    return lax.cond(
        padding_cell,
        lambda: Cell(
            lower_bound=zeros_like(lb),
            upper_bound=zeros_like(ub),
            broken=asarray(False),
            padding=asarray(True),
        ),
        lambda: cell,
    )


def _inside_adf(adf, cell, *args, **kwargs):
    lb, ub, _, _ = cell
    c = (lb + ub) / 2
    return adf(c, *args, **kwargs) >= 0


@partial(jit, static_argnames=("adf",))
def _split_cell(adf: ADF, lb, ub, *args: Any, eps=1e-6, **kwargs: Any) -> Cell:
    split_point = (lb + ub) / 2
    lower_bounds, upper_bounds, broken_cells, padding_cells = [], [], [], []
    for split in itertools.product([0, 1], repeat=len(lb)):
        new_lower = where(array(split) == 0, lb, split_point)
        new_upper = where(array(split) == 0, split_point, ub)
        broken_cell, padding_cell = _broken_or_padding_cell(adf, new_lower, new_upper, *args, eps=eps, **kwargs)
        lower_bounds.append(new_lower)
        upper_bounds.append(new_upper)
        broken_cells.append(broken_cell)
        padding_cells.append(padding_cell)

    lower_bounds = asarray(lower_bounds)
    upper_bounds = asarray(upper_bounds)
    broken_cells = asarray(broken_cells)
    padding_cells = asarray(padding_cells)
    return Cell(lower_bounds, upper_bounds, broken_cells, padding_cells)


_BrokenCellMask: TypeAlias = Array
_PaddingMask: TypeAlias = Array


@partial(jit, static_argnames=("adf",))
def _broken_or_padding_cell(adf: ADF, lb, ub, *args: Any, eps=1e-6, **kwargs: Any) -> tuple[_BrokenCellMask, _PaddingMask]:
    support = jnp.prod((ub - lb) / 2)
    assert lb.shape[0] == ub.shape[0]
    dim = lb.shape[0]
    cell_domain = jnp.stack(jnp.meshgrid(*[jnp.array([l, u]) for l, u in zip(lb, ub)]), axis=-1)
    center = (lb + ub) / 2

    LD = jnp.apply_along_axis(adf, -1, cell_domain, *args, **kwargs)
    LX = adf(center)
    max_count = 1 + 2**dim
    nD = jnp.sum(LD >= 0 - eps)
    nX = asarray((LX >= (0 - eps))).astype(nD.dtype)
    n = nD + nX
    padding_cell = ((n == 0) | ((nX == 0) & jnp.all(LD <= 0 + eps))) | (support == 0)
    broken_cell = (n != max_count) & (n > 0)
    broken_cell = asarray(jnp.where(padding_cell, False, broken_cell))
    return broken_cell, padding_cell
