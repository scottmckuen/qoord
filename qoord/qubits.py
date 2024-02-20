from copy import deepcopy

from qoord.__support__ import closest
from qoord.states import MatrixOperator, QuantumState


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
    def __init__(self, state: QuantumState, label: int | str = 0):
        self._quantum_state = state
        self._state_index = label

    def get_state(self, force_density_matrix=False):
        inner_state = self._quantum_state.get_value(force_density_matrix)
        return deepcopy(inner_state)

    def apply(self, operator: MatrixOperator) -> None:
        state = self._quantum_state
        state.apply(operator, [self._state_index])

    def measure(self, observable:  MatrixOperator):
        eigenvalue = self._quantum_state.measure(observable, self._state_index)
        return eigenvalue


class QubitSet(object):

    def __init__(self, qubits: list[Qubit]):
        q1 = qubits[0]
        self._global_state = q1._quantum_state
        self._state_indexes = [q._state_index for q in qubits]

    def __getitem__(self, item):
        return Qubit(self._global_state, item)

    def apply(self, operator: MatrixOperator) -> None:
        state = self._global_state
        state.apply(operator, self._state_indexes)

    def measure(self, observable:  MatrixOperator):
        eigenvalue = self._global_state.measure(observable, on_qubits=self._state_indexes)
        return eigenvalue

    def __iter__(self):
        for idx in self._state_indexes:
            yield Qubit(self._global_state, idx)


def binary_from_measurement(observable, qubit):
    """
    Map the two eigenvalues of the observable to binary values 0, 1,
    then measure the provided qubit to get a binary result.  This is
    helpful when interpreting measurements as choices or actions.

    @param observable:  essentially, what basis we choose to measure in
    @param qubit:  which qubit we are measuring
    @return: 0 if we measure the first eigenvalue, 1 if the second
    """
    op = observable.to_operator()
    values, vectors = op.eig()
    actions = {v: a for v, a in zip(values, (0, 1))}
    measurement = qubit.measure(op)
    measurement_corrected = closest(measurement, values)
    chosen_action = actions[measurement_corrected]
    return chosen_action
