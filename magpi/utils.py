from typing import Callable, TypeVar

from .prelude import *
import chex

# some function taken from jaxopt:
T = TypeVar("T", bound=Callable[..., chex.ArrayTree])

def make_funs_with_aux(fun: Callable, value_and_grad: bool, has_aux: bool):
    if value_and_grad:
        # Case when `fun` is a user-provided `value_and_grad`.

        if has_aux:
            fun_ = lambda *a, **kw: fun(*a, **kw)[0]
            value_and_grad_fun = fun
        else:
            fun_ = lambda *a, **kw: (fun(*a, **kw)[0], None)

            def value_and_grad_fun(*a, **kw):
                v, g = fun(*a, **kw)
                return (v, None), g

    else:
        # Case when `fun` is just a scalar-valued function.
        if has_aux:
            fun_ = fun
        else:
            fun_ = lambda p, *a, **kw: (fun(p, *a, **kw), None)

        value_and_grad_fun = jax.value_and_grad(fun_, has_aux=True)

    def grad_fun(*a, **kw):
        (v, a), g = value_and_grad_fun(*a, **kw)
        return g, a

    return fun_, grad_fun, value_and_grad_fun


def tree_single_dtype(tree):
    """The dtype for all values in e tree."""
    dtypes = set(p.dtype for p in tree_leaves(tree) if isinstance(p, Array))
    if not dtypes:
        return None
    if len(dtypes) == 1:
        return dtypes.pop()
    raise ValueError(f"Found more than one dtype in the tree ({list(dtypes)}).")


def tree_dim(tree: chex.ArrayTree, axis=0) -> int:
    dim = tree_leaves(tree_map(lambda t: t.shape[axis], tree))
    dim = set(dim)
    assert (
        len(dim) == 1
    ), f"Dimension mismatch! All arrays must have same size along axis {axis}."
    return dim.pop()


def apply_along_last_dims(
    func: Callable[..., T], *arr: chex.ArrayTree, dims=1
) -> T:
    """Uses vmap to apply a function over all axes of an array
    except the last ones specified by `dims`.

    Parameters
    ----------
    func1d : Callable
    *args : ArrayTree
    dims : int, optional
        The number of remaining axes. If 0 the function is
        applied on each scalar of the array. Default is 1

    Returns
    -------
    ArrayTree
    """
    _dims = [a.ndim for a in tree.leaves(arr)]
    max_dims = max(_dims)
    vmap_dims = max_dims - dims

    for axis in range(0, vmap_dims):
        func = jax.vmap(func, in_axes=axis, out_axes=axis)

    return func(*arr)
