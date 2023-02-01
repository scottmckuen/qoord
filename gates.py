from qoord.__support__ import MatrixArray
from qoord.states import MatrixOperator
from qoord.qubits import Qubit, QubitSet


class UnitaryGate:
    def __init__(self, matrix: MatrixArray | MatrixOperator):

        if isinstance(matrix, MatrixOperator):
            self._matrix = matrix
        else:
            self._matrix = MatrixOperator(matrix)

        if not self._matrix.is_unitary():
            raise ValueError(f"Input operator {matrix} is not unitary.")

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




