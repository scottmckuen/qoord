import cmath
import math

from copy import deepcopy
from numbers import Number

import numpy as np

"""
This is a lowest-level module that can safely be
imported by anything else in the package without
creating circular dependencies.

DO NOT import any other part of qoord into this module.

"""


def tupleize(array):
    new_array = [tuple(r) for r in array]
    return tuple(new_array)


def eipn(n: int) -> complex:
    """
    Shorthand for e^(i*pi/n)
    @param n: an integer
    @return: the math expression above
    """
    return cmath.exp(1j*math.pi/n)


def close_enough(array1, array2, rel_tol=1e-15):
    if isinstance(array1, np.ndarray):
        array1 = [x for x in array1.flat]
    if isinstance(array2, np.ndarray):
        array2 = [x for x in array2.flat]
    compare = zip(array1, array2)
    are_close = [cmath.isclose(x, y, rel_tol=rel_tol) for x, y in compare]
    result = all(are_close)
    return result


def int_to_binary_list(x: int, size: int) -> list[chr]:
    return list(f"{x:0{size}b}")


def binary_list_to_int(x: list[chr]) -> int:
    return int(''.join(x), base=2)


def update_index(x: int, permutation: dict, size: int) -> int:
    x = int_to_binary_list(x, size)
    new_x = deepcopy(x)
    for old_idx, new_idx in permutation.items():
        # Instead of breaking here, do you actually want to assert or raise?
        if old_idx >= len(x):
            raise ValueError(f"Permutation is wrong length? {len(x)}, {old_idx}")
        new_x[new_idx] = x[old_idx]

    new_x = binary_list_to_int(new_x)
    return new_x


def ndim_zero_ket(n_qubits: int) -> tuple[Number]:
    ket = 2**n_qubits * [0]
    ket[0] = 1
    return ket


def is_square(array):
    for row in array:
        if len(row) != len(array):
            return False
    
    return len(array) > 0


def closest(val, items, tolerance=1e-15):
    """
    Sometimes floating-point issues cause a measured value to
    be slightly different from the expected value.  If we have a
    set of known target values, this picks the closest one,
    within some tolerance
    @param val:  the candidate value that should be in the list of items
    @param items:  allowed set of values
    @param tolerance:  maximum allowed discrepancy
    @return: the entry from items that is closest, within the tolerance
    """
    gap = np.inf
    best_val = None
    for item in items:
        delta = item - val
        d2 = delta * np.conj(delta)  # could be complex-valued
        if d2 > tolerance:
            continue
        if d2 < gap:
            gap = d2
            best_val = item

    if best_val is None:
        msg = f"Could not find match for {val} within tolerance {tolerance}"
        raise ValueError(msg)
    return best_val
