from .gates import UnitaryGate
from .core_operators import *

Identity = UnitaryGate(identity_op).__call__
PauliX = UnitaryGate(pauli_x).__call__
PauliY = UnitaryGate(pauli_y).__call__
PauliZ = UnitaryGate(pauli_z).__call__
Hadamard = UnitaryGate(hadamard_op).__call__
S = UnitaryGate(phase_op).__call__
T = UnitaryGate(pi_over_8_gate_op).__call__
CNOT = UnitaryGate(cnot_op).__call__
