from typing import TypeVar

from .states import MatrixArray, MatrixOperator
from .qubits import Qubit, QubitSet

TmpUG = TypeVar("TmpUG", bound="UnitaryGate")


class UnitaryGate:
    def __init__(self, matrix: MatrixArray | MatrixOperator, name=None):

        if isinstance(matrix, MatrixOperator):
            self._matrix = matrix
        else:
            self._matrix = MatrixOperator(matrix)

        if not self._matrix.is_unitary():
            raise ValueError(f"Input operator {matrix} is not unitary.")

        self._name = name

    def __repr__(self):
        return f'{self._name}'

    def __call__(self, qubit: Qubit, *qubits: Qubit):
        """
        The gate call with the qubit argument is implemented using the
        qubit.apply(matrix operator) method.

        @param qubit:
        @param qubits:
        @return:
        """
        if qubits:
            qubits = [qubit] + [x for x in qubits]
            qubits = QubitSet(qubits)
        else:
            qubits = qubit
        qubits.apply(self._matrix)

    def __pow__(self, exponent):
        new_matrix = self._matrix.__pow__(exponent)
        return UnitaryGate(new_matrix, f'{self}^{exponent}')

    def __neg__(self):
        new_matrix = -self._matrix
        return UnitaryGate(new_matrix, f'-{self}')

    def to_operator(self):
        return self._matrix

    def tensor(self, other: TmpUG) -> TmpUG:
        op1 = self.to_operator()
        op2 = other.to_operator()
        new_op = op1.tensor(op2)
        return UnitaryGate(matrix=new_op, name=f'{self}x{other}')
