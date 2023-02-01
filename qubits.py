from copy import deepcopy

from .states import MatrixOperator, QuantumState


class Qubit(object):
    """
    This is the basic resource that can be used for computing
    and coordination.  The global state is shared among several
    qubits, and the label tracks which tensor component corresponds
    to this qubit.

    A Qubit acts like a handle to a particular part of the state, with
    a label for that part.  There is no other internal state besides
    the global quantum_state.  If two qubits have the same label and
    share the same quantum state, there's no problem with that.

    FIXME:  currently unsolved:  what happens if you create two Device objects,
    then use a binary gate operation like CNOT to entangle two qubits from
    different devices?  This needs to create a joint quantum state object
    but the devices would only have access to some of the state.

    """
    def __init__(self, state: QuantumState, label: int = 0):
        self._quantum_state = state
        self._state_index = label

    def get_state(self, force_density_matrix=False):
        inner_state = self._quantum_state.get_value(force_density_matrix)
        return deepcopy(inner_state)

    def apply(self, operator: MatrixOperator) -> None:
        state = self._quantum_state
        state.apply(operator, [self._state_index])

    def measure(self, observable:  MatrixOperator):
        result = self._quantum_state.measure(observable, self._state_index)
        return result


class QubitSet(object):
    def __init__(self, qubits: list[Qubit]):
        q1 = qubits[0]
        self._global_state = q1._quantum_state
        self._state_indexes = [q._state_index for q in qubits]

    def apply(self, operator: MatrixOperator) -> None:
        state = self._global_state
        state.apply(operator, self._state_indexes)

    def __iter__(self):
        for idx in self._state_indexes:
            yield Qubit(self._global_state, idx)


