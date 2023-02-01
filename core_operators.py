import math

from .__support__ import eipn
from .states import MatrixOperator, identity_op

pauli_x = MatrixOperator([[0, 1],
                          [1, 0]])

pauli_y = MatrixOperator([[0, -1j],
                          [1j, 0]])

pauli_z = MatrixOperator([[1, 0],
                          [0, -1]])

c = math.sqrt(1.0/2.0)
hadamard_op = MatrixOperator([[c,  c],
                              [c, -c]])

phase_op = MatrixOperator([[1, 0],
                           [0, 1j]])

pi_over_8_gate_op = MatrixOperator([[1, 0],
                                    [0, eipn(4)]])

cnot_op = MatrixOperator([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0]])

