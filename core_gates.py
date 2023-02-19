from .states import identity_op
from .gates import UnitaryGate
from .core_operators import *

Identity = UnitaryGate(identity_op)
PauliX = UnitaryGate(pauli_x)
PauliY = UnitaryGate(pauli_y)
PauliZ = UnitaryGate(pauli_z)
Hadamard = UnitaryGate(hadamard_op)
S = UnitaryGate(phase_op)
T = UnitaryGate(pi_over_8_gate_op)
CNOT = UnitaryGate(cnot_op)
