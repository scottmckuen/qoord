from .states import identity_op
from .gates import UnitaryGate
from .core_operators import *

Identity = UnitaryGate(identity_op, 'I')
PauliX = UnitaryGate(pauli_x, 'X')
PauliY = UnitaryGate(pauli_y, 'Y')
PauliZ = UnitaryGate(pauli_z, 'Z')
Hadamard = UnitaryGate(hadamard_op, 'H')
S = UnitaryGate(phase_op, 'S')
T = UnitaryGate(pi_over_8_gate_op, 'T')
CNOT = UnitaryGate(cnot_op, 'CNOT')
