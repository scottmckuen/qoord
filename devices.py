import copy
import typing

from qoord.__support__ import ndim_zero_ket
from qoord.qubits import Qubit, QubitSet
from qoord.core_gates import Hadamard, CNOT
from qoord.states import DensityMatrix, StateVector, QuantumState


class Device(object):
    """
    A device enables *local* manipulation of a subset
    of the available qubits in the global system.

    An n-qubit device defaults to an initial state of |0>^n.
    """
    def __init__(self, qubits: int | list[str]):
        # assign unique identifiers to the qubits
        if isinstance(qubits, int):
            self._num_qubits = qubits
            self._qubit_ids = [_ for _ in range(qubits)]
        else:
            self._num_qubits = len(qubits)
            self._qubit_ids = qubits

        default_init_state = ndim_zero_ket(qubits)
        state = QuantumState(self._qubit_ids)
        state.set_value(StateVector(default_init_state))
        self._quantum_state = state
        self._original_qubit_ids = copy.deepcopy(self._qubit_ids)

    def initialize(self, state: StateVector | DensityMatrix):
        """
        This DOES NOT preserve the input state vector, it copies
        the contents into the shared internal state of the device,
        which are also held by the qubits
        """
        state = copy.deepcopy(state)
        self._quantum_state.set_value(state)

    def get_state(self):
        inner_state = self._quantum_state.get_value()
        return copy.deepcopy(inner_state)

    def get_qubit(self, key: typing.Hashable) -> Qubit:
        state = self._quantum_state
        result = Qubit(state, label=key)
        return result

    def get_qubits(self, keys: [typing.Hashable, list[typing.Hashable]]) -> Qubit | QubitSet:
        singleton = False
        if isinstance(keys, typing.Hashable):
            keys = [keys]  # because it's a singleton
            singleton = True

        state = self._quantum_state
        if singleton:
            results = Qubit(state, label=keys[0])
        else:
            qubits = [Qubit(state, label=k) for k in keys]
            results = QubitSet(qubits)

        return results

    def make_bell_pair(self, qubits):
        q1, q2 = self.get_qubits(qubits)
        Hadamard(q1)  # create superposition
        CNOT(q1, q2)  # entangle first shared pair
        return QubitSet([q1, q2])

