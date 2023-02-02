import cmath
import contextlib
import copy
import math
import random
import typing

import numpy as np  # best if we don't need this!

from functools import partial
from typing import TypeAlias, TypeVar, Union

from .__support__ import Numeric, MatrixArray, VectorArray, ndim_zero_ket, update_index, is_square

# TmpMO exists because "Self" isn't available until Python 3.11
TmpSV = TypeVar("TmpSV", bound="StateVector")
TmpMO = TypeVar("TmpMO", bound="MatrixOperator")
TmpDM = TypeVar("TmpDM", bound="DensityMatrix")

InternalArray: TypeAlias = Union[VectorArray, np.array]


def _copy_array(a):
    new_a = copy.deepcopy(a)
    try:
        new_a = new_a.tolist()
    except:
        pass
    return new_a


class StateVector(object):

    @classmethod
    def _normalize(cls, array):
        # need to make sure array norms to 1
        norm = math.sqrt(sum([x * x for x in array]))
        if norm != 1:
            normed_array = [x / norm for x in array]
        else:
            normed_array = array

        return normed_array

    def __init__(self, array: VectorArray):
        """
        Contract:  self._array is always a (nested?) list,
        and we only use np.arrays to assist with calculations
        or as a return value when that's clearly expected
        from the API.

        The StateVector is shareable, but the input array
        shouldn't be.  It is a numeric array, so we don't
        need to deepcopy.
        """
        if array is None:  # does type hinting help prevent this?
            raise ValueError("Cannot make StateVector with no array input.")

        _array = self._normalize(array)
        self._array = _array

    """def update(self, new_array: InternalArray):
        if isinstance(new_array, np.ndarray):
            new_array = new_array.tolist()
        self._array = [x for x in new_array]"""

    def to_array(self):
        return self._array

    def to_numpy_array(self):
        return np.array(self.to_array())

    def __repr__(self):
        return str(self._array)  # we just care about the contents

    def __eq__(self, other: TmpSV):
        if not isinstance(other, StateVector):
            # what about some other representation of state?  Later....
            return False
        return np.array_equal(self._array, other._array)

    def __len__(self):
        return len(self._array)

    def __mul__(self, other):
        vector = self.to_numpy_array()
        vector = vector * other
        return StateVector(array=vector)

    def __truediv__(self, other):
        vector = self.to_numpy_array()
        vector = vector / other
        return StateVector(array=vector)

    def qubit_count(self):
        return int(math.log(len(self), 2))

    def tensor(self, other: TmpSV):
        array_out = np.tensordot(self._array, other._array, axes=0)
        array_out = array_out.reshape([array_out.size, ])
        array_out = [x for x in array_out]
        return StateVector(array=array_out)

    def dot(self, other: TmpSV):
        oparts = other._array
        self_parts = self._array
        parts = zip(self_parts, oparts)
        parts = [x * y for x, y in parts]
        result = sum(parts)
        return result

    def to_density_matrix(self) -> TmpDM:
        content = np.outer(self.to_array(), self.to_array())
        return DensityMatrix(content)

    def rearrange(self, tensor_permutation):
        new_inner = rearrange_vector(tensor_permutation, self._array)
        return StateVector(new_inner)


class DensityMatrix(object):
    def __init__(self, array: MatrixArray | VectorArray):
        """
        Contract:  self._array is always a (nested?) list,
        and we only use np.arrays to assist with calculations
        or as a return value when that's clearly expected
        from the API.
        """
        array = _copy_array(array)
        if isinstance(array[0], list):
            assert is_square(array)
            self._array = array
        else:  # hopefully scalar
            self._array = np.outer(array, array).tolist()

    def update(self, new_array: MatrixArray | VectorArray | StateVector):

        assert new_array.shape == self._array.shape
        self._array = _copy_array(new_array)

    def to_array(self):
        return self._array

    def to_numpy_array(self):
        return np.array(self._array)

    def __repr__(self):
        return str(self._array)

    def __eq__(self, other: TmpDM):
        return np.array_equal(self._array, other._array)

    def tensor(self, other: TmpDM):
        array_out = np.kron(self._array, other._array)
        return DensityMatrix(array=array_out)

    def trace_out(self, keep_qubits: list) -> TmpDM:
        pass


class MatrixOperator(object):
    @classmethod
    def lift(cls, function: typing.Callable) -> typing.Callable:
        """
        Convert any one-variable function on complex numbers into
        a compatible function on normal operators.  Do this by
        diagonalizing using the eigenvector structure, applying
        the function to the eigenvalues, then building the new
        operator using a basis of outer products of the eigenvectors.
        @param function: 1-variable complex to complex (not checked!)
        @return:  new MatrixOperator
        """
        def new_func(mo: MatrixOperator) -> MatrixOperator:
            values, vectors = mo.eig()
            basis = [np.outer(v, v) for v in vectors]
            new_vals = [function(val) for val in values]
            matrix_new = sum([val * vec for val, vec in zip(new_vals, basis)])
            mo_new = MatrixOperator(matrix_new)
            return mo_new
        return new_func

    def __init__(self, components: MatrixArray):
        # FIXME: should we do the same thing as SV and DM
        # and coerce this to a list of lists?
        self._array = np.array(components)

    def __eq__(self, other: TmpMO):
        self_array = self.to_array()
        other_array = other.to_array()
        return np.array_equal(self_array, other_array)

    def __matmul__(self, other: TmpMO | StateVector):
        if isinstance(other, MatrixOperator):
            new_array = self._array @ other._array
            return MatrixOperator(components=new_array)
        else:  # StateVector
            return self.apply_to_vector(other)

    def apply_to_vector(self, vector: StateVector):
        vector_array = vector.to_numpy_array()
        new_vector = self._array @ vector_array
        return StateVector(array = new_vector)

    def __pow__(self, exponent):
        def cpower(a, b):
            a = complex(a)
            b = complex(b)
            return a**b
        power_func = self.lift(lambda x: cpower(x, exponent))  # does this get slow?
        return power_func(self)

    def tensor(self, other: TmpMO):
        array_out = np.kron(self._array, other._array)
        return MatrixOperator(components=array_out)

    def to_array(self, as_numpy=False):
        if as_numpy:
            return self._array.copy()
        else:
            return self._array.tolist()

    def is_unitary(self):
        m = self._array
        return np.allclose(np.eye(len(m)), m.dot(m.T.conj()))

    def dim(self):
        """
        The size of the operator as a square array
        @return:
        """
        return self._array.shape[0]

    def qubit_count(self):
        return int(math.log(self.dim(), 2))

    def eig(self) -> tuple[list[Numeric], list[VectorArray]]:
        """
        Get the eigenvalue, eigenvector structure of the operator.
        Improves on np.linalg.eig by returning eigenvectors as
        a list, rather than a numpy array, for clarity

        @return: values, vectors
            - values:  List[numeric]
            - vectors: List[numpy vectors]
        """
        values, vectors = np.linalg.eig(self._array)  # maybe cache this?
        n_dims = len(values)
        vectors = [vectors[:, i] for i in range(n_dims)]
        """
        WARNING:  "vectors.tolist()" does not work in the above line 
        because the matrix of eigenvectors from .eig() has an unexpected 
        orientation:  you will get a meaningless set of vectors!
        """

        return values, vectors

    def eigenvalues(self):
        values, _ = self.eig()
        return values

    def eigenvectors(self):
        _, vectors = self.eig()
        return vectors

    def get_eigenvector(self, eigenvalue):
        values, vectors = self.eig()
        idx = values.index(eigenvalue)
        return vectors[idx]

    def distribution(self, state: StateVector | DensityMatrix):
        eigenvalues, eigenstates = self.eig()
        eigenstates = [StateVector(s) for s in eigenstates]
        coefficients = [state.dot(s) for s in eigenstates]
        return eigenvalues, eigenstates, coefficients

    def measure(self, state: StateVector | DensityMatrix, extra_data=False) \
            -> float | tuple[float, TmpSV, float]:
        eigenvalues, eigenstates, coefficients = self.distribution(state)
        probabilities = [c*c for c in coefficients]

        n_choices = len(eigenvalues)
        select = random.choices(range(n_choices), probabilities, k=1)
        result_idx = select[0]

        value = eigenvalues[result_idx]
        vector = eigenstates[result_idx]
        p = probabilities[result_idx]

        if extra_data:
            return value, vector, p
        else:
            return value

    def __repr__(self):
        return str(self._array)

    def expand_with_identities(self, num_dims: int):
        """
        The point of this function is to elevate this operator to a
        higher-dimensional space by adding dummy dimensions.  This
        helps us operate on one part of the total quantum state of
        the system while leaving the rest unchanged.

        Return a new operator on k+n dimensions, where k is the number
        of added dimensions, and n is dim(self).  The new operator
        is a tensor product of k Identity operators followed by the
        current operator, so the active dimensions are always at the end.

        @param num_dims: number of new "identity" dimensions to add.
        @return: a new operator on num_dims + self.dim() dimensions
        """
        op_parts = []
        for _ in range(num_dims):
            op_parts.append(identity_op)
        op_parts.append(self)  # current operator is last
        new_op = op_parts[0]
        for op in op_parts[1:]:
            new_op = new_op.tensor(op)
        return new_op


identity_op = MatrixOperator([[1, 0],
                              [0, 1]])


def rearrange_vector(tensor_permutation: map, state_vector: list, size=None):
    if not size:
        size = len(state_vector).bit_length() - 1
        # this will probably be wrong if the vector is not 2^n length

    new_state = copy.deepcopy(state_vector)

    for idx in range(len(new_state)):
        new_idx = update_index(idx, tensor_permutation, size)
        new_state[new_idx] = state_vector[idx]

    return new_state


def permute_to_end(move_these: list, total_set: list):
    for q_val in move_these:
        qi = total_set.index(q_val)
        total_set.pop(qi)  # take q_val out
        total_set.append(q_val)  # put q_val back

    perm = {v: idx for idx, v in enumerate(total_set)}
    return perm


def invert_permutation(permutation):
    new_perm = {v: k for k, v in permutation.items()}
    return new_perm


class QuantumState(object):

    def __init__(self, qubit_ids):
        self.qubit_ids = qubit_ids
        z = ndim_zero_ket(len(qubit_ids))
        self._state_vector = z
        self._density_matrix = None

    def __repr__(self):
        if self._density_matrix is None:
            return str(self._state_vector)
        else:
            return str(self._density_matrix)

    def size(self) -> int:
        return 2**self.qubit_count()

    def qubit_count(self):
        return len(self.qubit_ids)

    def get_value(self, force_density_matrix=False) -> TmpDM | TmpSV:
        if self._density_matrix is None:
            value = self._state_vector
            if force_density_matrix:
                value = value.to_density_matrix()
        else:
            value = self._density_matrix
        return value

    def set_value(self, state: TmpSV | TmpDM) -> None:
        if len(state) != self.size():
            raise ValueError(f"Input state must be of size {self.size()}")

        if isinstance(state, StateVector):
            self._state_vector = state
            self._density_matrix = None
        elif isinstance(state, DensityMatrix):
            self._state_vector = None
            self._density_matrix = state

    @contextlib.contextmanager
    def _align_to_end(self, operator: TmpMO, on_qubits: list):
        """
        tensor an operator with a bunch of identities so it operates
        on the tail of the internal qubit list; permute the global state
        to put the key qubits at the end, in the order listed; then
        yield to whatever needs to be done; finally permute the state back
        """
        # 1) tensor the operator with a bunch of identities
        gate_qubits = operator.dim().bit_length() - 1
        n_qubits = self.qubit_count()
        n_identities = n_qubits - gate_qubits
        new_op = operator.expand_with_identities(n_identities)

        # permute the global state to put the key qubits at the end
        perm = permute_to_end(move_these=on_qubits, total_set=self.qubit_ids)

        v = self.get_value()  # is the copy needed?
        v = v.rearrange(tensor_permutation=perm)
        self.set_value(v)

        yield new_op, perm

        # have to re-fetch the state because the external context can modify it
        # maybe this is a case for using yield-from?

        v = self.get_value()

        # permute the state back
        inv_perm = invert_permutation(perm)
        v = v.rearrange(inv_perm)
        self.set_value(v)

    def apply(self, operator: TmpMO, on_qubits: list) -> None:
        with self._align_to_end(operator, on_qubits) as ctxt:
            # apply the full state operator
            new_op, perm  = ctxt
            v = self.get_value()
            v = new_op @ v
            self.set_value(v)

    def measure(self, observable: MatrixOperator, on_qubits: list | None = None):
        if on_qubits is None and \
                observable.qubit_count() == self.qubit_count():
            on_qubits = self.qubit_ids

        with self._align_to_end(observable, on_qubits) as ctxt:
            # Perform the measurement
            new_op, perm = ctxt
            state = self.get_value()
            result, vector, prob = new_op.measure(state, extra_data=True)

            # update the global quantum state to account for the measurement
            # and do this in-context because the shuffle needs to be applied
            dm = vector.to_density_matrix()
            projector = MatrixOperator(dm.to_array())
            updated_state = (projector @ state) / math.sqrt(prob)

            self.set_value(updated_state)

            # some things we have to do manually
            inv_perm = invert_permutation(perm)
            vector = vector.rearrange(inv_perm)

        # reduce the measurement result to the user-visible qubits
        if not set(on_qubits) == set(self.qubit_ids):
            vector = vector.partial_trace(on_qubits)
        values, vectors = observable.eig()
        for vidx, v in enumerate(vectors):
            if all(v == vector):
                break
        result = values[vidx]

        return result