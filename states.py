import contextlib
import math
import random
import typing

import numpy as np

from numbers import Number
from typing import TypeAlias, TypeVar, Union

from .__support__ import Numeric, ndim_zero_ket, update_index, is_square, _copy_array

# row and column vectors are both supported
RowVector: TypeAlias = tuple[Numeric]
ColumnVector: TypeAlias = tuple[tuple[Numeric]]
VectorArray: TypeAlias = tuple[Numeric] | tuple[tuple[Numeric]]
MatrixArray: TypeAlias = tuple[RowVector]

# TmpMO exists because "Self"-typing isn't available until Python 3.11
TmpSV = TypeVar("TmpSV", bound="StateVector")
TmpMO = TypeVar("TmpMO", bound="MatrixOperator")
TmpDM = TypeVar("TmpDM", bound="DensityMatrix")


# data like [1, 2, 3]
def _flat(data):
    return all([isinstance(x, Number) for x in data])


# data like [[1, 2, 3]]
def _np_row(data):
    one_row = len(data) == 1
    tuple_or_list = isinstance(data[0], tuple) or isinstance(data[0], list)
    numeric = all([isinstance(x, Number) for x in data[0]])
    return one_row and tuple_or_list and numeric


# data like [[1], [2], [3]]
def _np_column(data):
    return all([len(x) == 1 and isinstance(x[0], Number) for x in data])


class StateVector(object):

    @classmethod
    def _normalize(cls, array):
        # need to make sure array norms to 1
        norm = math.sqrt(sum([x * x for x in array]))
        if norm != 1:
            normed_array = [x / norm for x in array]
        else:
            normed_array = array

        return tuple(normed_array)

    def __init__(self, array: VectorArray, as_ket=True):
        """
        Contract:  self._array is always a list,
        and we only use numpy arrays internally to assist
        with calculations or as a return value when that's
        clearly expected from the API.

        The StateVector is immutable and shareable, but the
        input array shouldn't be.  It is a numeric array, so
        we don't need to deepcopy.

        @param array:  the elements of the vector, as tuple or list
        @param as_ket:  flag for column (ket) vector or row (bra) vector; default True
        """
        if array is None:  # does type hinting help prevent this?
            raise ValueError("Cannot make StateVector with no array input.")

        if not isinstance(array, list) and not isinstance(array, tuple):
            raise ValueError("Only support list and tuple as input.")

        if _flat(array):
            self._array = self._normalize(array)
            self._as_ket = as_ket
        else:
            vtype, array = self._vector_type(array)
            self._as_ket = vtype == 'column'
            self._array = self._normalize(array)


    @classmethod
    def _vector_type(cls, data):
        # assume the data is already a list or tuple
        if _flat(data):
            vtype = 'row'
            flat_data = data
        elif _np_row(data):
            vtype = 'row'
            flat_data = data[0]
        elif _np_column(data):
            vtype = 'column'
            flat_data = [x[0] for x in data]
        else:
            raise ValueError("Data is the wrong shape!")

        return vtype, flat_data

    def to_array(self):
        return [x for x in self._array]

    def to_numpy_array(self):
        if self._as_ket:
            vtype = 'c'  # kets are column vectors
        else:
            vtype = 'r'  # bras are row vectors
        return np.r_[vtype, self._array].A


    def __repr__(self):
        return str(self._array)  # we just care about the contents

    def __eq__(self, other: TmpSV):
        if not isinstance(other, StateVector):
            return False
        return np.array_equal(self._array, other._array)

    def __hash__(self):
        return hash(self._array)

    def __len__(self):
        return len(self._array)

    def qubit_count(self):
        return int(math.log(len(self), 2))

    def tensor(self, other: TmpSV) -> TmpSV:
        array_out = np.tensordot(self._array, other._array, axes=0)
        array_out = array_out.reshape([array_out.size, ])
        array_out = [x for x in array_out]
        result = StateVector(array_out)
        return result

    def outer(self, other: TmpSV) -> TmpMO:
        content = np.outer(self.to_numpy_array(), other.to_numpy_array())
        return MatrixOperator(content)

    def adjoint(self):
        new_array = self.to_numpy_array().conj().T
        return StateVector(new_array.tolist())

    def dot(self, other: TmpSV) -> Numeric:
        other_parts = other.to_array()
        self_parts = self.to_array()
        parts = zip(self_parts, other_parts)
        parts = [x * y for x, y in parts]
        result = sum(parts)
        return result

    def to_density_matrix(self) -> TmpDM:
        content = self.outer(self)
        return DensityMatrix(content.to_array())

    def rearrange(self, tensor_permutation):
        new_inner = rearrange_vector(tensor_permutation, self._array)
        return StateVector(new_inner)


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

    def __matmul__(self, other: TmpMO | StateVector | np.array):
        if isinstance(other, MatrixOperator) or isinstance(other, DensityMatrix):
            self_array = self.to_array(as_numpy=True)
            other_array = other.to_array(as_numpy=True)
            new_array = self_array @ other_array
            return MatrixOperator(components=new_array)
        elif isinstance(other, StateVector):  # StateVector
            return self.apply_to_vector(other)
        elif isinstance(other, np.ndarray):
            self_array = self.to_array(as_numpy=True)
            new_array = self_array @ other
            if 1 not in other.shape:  # not a vector
                return MatrixOperator(new_array)  # this choice seems dodgy as fuck
            else:
                return new_array


    def apply_to_vector(self, vector: StateVector):
        vector_array = vector.to_numpy_array()
        new_vector = self.to_array(as_numpy=True) @ vector_array
        return StateVector(array=new_vector.tolist())

    def __pow__(self, exponent):
        def cpower(a, b):
            a = complex(a)
            b = complex(b)
            return a**b
        power_func = self.lift(lambda x: cpower(x, exponent))  # does this get slow?
        return power_func(self)

    def tensor(self, other: TmpMO | TmpSV):
        if isinstance(other, StateVector):
            array_out = np.kron(self._array, other.to_numpy_array())
        else:
            array_out = np.kron(self._array, other._array)
        return MatrixOperator(components=array_out)

    def tensor_power(self, num: int) -> TmpMO:
        result = self
        for _ in range(num - 1):
            result = result.tensor(self)
        return result

    def adjoint(self):
        new_array = self.to_array(as_numpy=True).conj().T
        return MatrixOperator(new_array)

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

    def shape(self):
        return self._array.shape

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

    def distribution(self, state: StateVector | TmpMO):
        eigenvalues, eigenstates = self.eig()
        eigenstates = [StateVector(s.tolist()) for s in eigenstates]
        coefficients = [state.dot(s) for s in eigenstates]
        return eigenvalues, eigenstates, coefficients

    def measure(self, state: StateVector | TmpMO, extra_data=False) \
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


class DensityMatrix(MatrixOperator):
    def __init__(self, array: MatrixArray | VectorArray):
        """
        Contract:  self._array is always a (nested?) list,
        and we only use np.arrays to assist with calculations
        or as a return value when that's clearly expected
        from the API.
        """
        if isinstance(array, typing.Sequence) and \
                not isinstance(array[0], typing.Sequence):
            # this is a vector input, so we want to cast it to a square array
            array = np.outer(array, array).tolist()
        super().__init__(array)

    def __eq__(self, other: TmpDM):
        return np.array_equal(self._array, other._array)

    def __add__(self, other: TmpDM):
        content = self.to_array(True) + other.to_array(True)
        return DensityMatrix(content)

    def scale(self, scale_factor):
        content = self.to_array(True) * scale_factor
        return DensityMatrix(content)

    def partial_trace(self, keep_qubits: list) -> TmpDM:
        """
        Compute a density operator that applies to a subset
         of the qubits.  The partial_trace operation effectively
         removes the influence of the other qubits by swapping
         in the expectation value of measuring them (I think).

        @param keep_qubits:
        @return:
        """

        qubit_ids = {x for x in range(self.qubit_count())}
        keep_qubits = set(keep_qubits)
        drop_qubits = qubit_ids.difference(keep_qubits)

        # To keep subsystem A (keep_qubits), and trace out
        # subsystem B (drop_qubits), we need a basis for the B-set,
        # and the identity on the A-set
        a_size, b_size = 2**len(keep_qubits), 2**len(drop_qubits)
        b_basis = np.identity(b_size)
        b_basis = [b_basis[:, i] for i in range(b_size)]
        a_identity = identity_op.tensor_power(len(keep_qubits))

        result = None
        #op = self.to_numpy_array()
        for j in b_basis:
            j = StateVector(j.tolist())
            jT = j.adjoint()
            left_side = a_identity.tensor(jT)
            right_side = a_identity.tensor(j)
            step_result = left_side.to_array(True) @ self.to_array(True)
            step_result = step_result @ right_side.to_array(True)
            if result is None:
                result = step_result
            else:
                result += step_result

        result = DensityMatrix(result)
        return result






def rearrange_vector(tensor_permutation: map, state_vector: list, size=None):
    if not size:
        size = len(state_vector).bit_length() - 1
        # this will probably be wrong if the vector is not 2^n length

    new_state = list(state_vector)

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
        with self._align_to_end(operator, on_qubits) as context:
            # apply the full state operator
            new_op, perm = context
            v = self.get_value()
            v = new_op @ v
            self.set_value(v)

    def measure(self, observable: MatrixOperator, on_qubits: list | None = None):
        if on_qubits is None:
            if observable.qubit_count() == self.qubit_count():
                on_qubits = self.qubit_ids
            else:
                raise ValueError("Specify the qubits to measure!")
        elif not isinstance(on_qubits, list) and not isinstance(on_qubits, set):
            on_qubits = [on_qubits]  # single qubit label of some kind?

        with self._align_to_end(observable, on_qubits) as context:

            # Perform the measurement
            new_op, perm = context
            state = self.get_value()
            evalue, evector, probability = new_op.measure(state, extra_data=True)

            # update the global quantum state to account for the measurement
            # and do this in-context because the shuffle needs to be applied
            projector = evector.to_density_matrix()  # because this uses the outer product
            updated_state = projector @ state.to_numpy_array()  # project to the chosen eigenvector
            updated_state = [s / math.sqrt(probability) for s in updated_state]
            updated_state = StateVector(updated_state)

            self.set_value(updated_state)

            # some things we have to do manually
            inv_perm = invert_permutation(perm)
            vector = evector.rearrange(inv_perm)  # return the eigenvector to the original arrangement

        """        
        # reduce the measurement result to the user-visible qubits
        current_state = self.get_value(force_density_matrix=True)
        if not set(on_qubits) == set(self.qubit_ids):
            current_state = current_state.partial_trace(keep_qubits=on_qubits)
            # FIXME:  what comes next here?

        values, vectors = observable.eig()
        for vidx, v in enumerate(vectors):
            if all(v == vector):
                break
        result = values[vidx]
        """
        return evalue