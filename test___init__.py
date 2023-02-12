import math

import numpy as np

from .__support__ import close_enough, update_index, tupleize
from .states import StateVector, DensityMatrix, MatrixOperator, QuantumState, permute_to_end
from .core_operators import identity_op, hadamard_op, cnot_op, pauli_z, pauli_x, phase_op
from .gates import UnitaryGate
from .core_gates import Hadamard, CNOT, PauliX
from .qubits import Qubit
from .devices import Device


def test_support_binary():
    shuffle = {3: 5, 5: 3}
    original = 1000
    expected = 952
    actual = update_index(original, shuffle, 9)
    assert actual == expected


def test_setup():
    device = Device(qubits=1)
    qubit = device.get_qubits(0)
    assert qubit is not None
    assert isinstance(qubit, Qubit)
    assert isinstance(device.get_state(), StateVector)

    device.initialize(StateVector((1, 0)))
    qubit_state_value = qubit.get_state()
    assert isinstance(qubit_state_value, StateVector)

    assert qubit.get_state()._array is not None


def test_two_qubits():
    device = Device(qubits=2)
    device.initialize(StateVector((1, 0, 0, 0)))

    qubit = device.get_qubits(1)
    assert isinstance(qubit.get_state(), StateVector)

    assert qubit.get_state()._array is not None


def test_matrix_operator():

    mo = MatrixOperator(((0, 1),
                         (1, 0)))

    sv = StateVector(array=(0.3, 0.8))

    expected = StateVector(array=(0.8, 0.3))
    actual = mo @ sv

    assert close_enough(actual.to_array(), expected.to_array())


def test_matrix_operator_dimension():
    mo = MatrixOperator(((0, 1),
                         (1, 0)))
    expected = 2
    actual = mo.dim()
    assert actual == expected


def test_qubit_apply():
    mo = MatrixOperator(((0, 1),
                         (1, 0)))

    sv = StateVector(array=(0.3, 0.8))
    state = QuantumState([0])
    state.set_value(sv)
    qb = Qubit(state=state, label=0)
    qb.apply(mo)

    expected = StateVector(array=(0.8, 0.3))
    actual = qb.get_state()

    assert close_enough(expected.to_array(), actual.to_array())


def test_permute_to_end():
    actual = permute_to_end([3, 1], [x for x in range(6)])
    expected = {0: 0, 1: 5, 2: 1, 3: 4, 4: 2, 5: 3}
    assert expected == actual


def test_superposition():

    device = Device(qubits=1)

    init_state = StateVector((1, 0))
    device.initialize(init_state)
    qubit = device.get_qubits(0)

    Hadamard(qubit)

    state = qubit.get_state()
    actual_array = state.to_array()

    expected_state = StateVector((1, 1))
    expected_array = expected_state.to_array()
    assert close_enough(actual_array, expected_array)


def test_cnot():
    device = Device(qubits=2)
    device.initialize(StateVector((0, 0, 1, 0)))

    qubits = device.get_qubits([0, 1])
    CNOT(qubits)

    expected = StateVector((0, 0, 0, 1))
    actual = device.get_state()

    assert close_enough(actual.to_array(), expected.to_array())


def test_is_unitary_validation():
    """ Does UnitaryGate check for unitary input?"""
    try:
        foo = ((1, 0), (0, 2))  # not unitary, deliberately
        u = UnitaryGate(foo)
        assert False
    except:
        assert True


def test_bell_pair():

    init_state = StateVector((1, 0, 0, 0))

    hadamard_2qb = hadamard_op.tensor(identity_op)
    state2 = hadamard_2qb @ init_state
    state3 = cnot_op @ state2

    expected_state = StateVector((math.sqrt(1/2), 0, 0, math.sqrt(1/2)))
    assert close_enough(expected_state._array, state3._array)


def test_state_vector_to_density_matrix():
    sv = StateVector((1, 0))
    actual = sv.to_density_matrix()
    expected = DensityMatrix(((1, 0), (0, 0)))

    assert actual == expected


def test_bell_pair_measurement():
    n_runs = 1000
    success = 0
    for _ in range(n_runs):
        bell_state = StateVector((math.sqrt(1/2), 0, 0, math.sqrt(1/2)))
        quantum_state = QuantumState((0, 1))
        quantum_state.set_value(bell_state)
        bob_op = pauli_z  # measure in the lab basis
        ev = quantum_state.measure(bob_op, [1])

        # now allow alice to measure - she should see the same eigenvalue every time
        alice_op = pauli_z
        ev2 = quantum_state.measure(alice_op, [0])

        if ev2 == ev:
            success += 1

    assert success == n_runs


def test_tensor_sv():
    sv1 = StateVector((1, 2))
    sv2 = StateVector((3, 4))

    sv3 = sv1.tensor(sv2)
    actual = sv3.to_array()
    expected = StateVector._normalize((3, 4, 6, 8))

    assert close_enough(actual, expected)


def test_density_matrix():
    m = (1, 0)
    sv1 = DensityMatrix(m)

    a1_actual = sv1.to_array()
    a1_expected = tupleize(np.outer(m, m).tolist())
    assert a1_actual == a1_expected

    sv2 = DensityMatrix((1/2, 1/2))

    sv3 = sv1.tensor(sv2)
    actual = sv3.to_array()
    expected = ((0.25, 0.25, 0, 0),
                (0.25, 0.25, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0))

    assert actual == expected


def test_tensor_mo():
    mo1 = MatrixOperator(((1, 0), (0, -1)))
    mo2 = MatrixOperator(((1, 0), (0, 1)))

    mo3 = mo1.tensor(mo2)
    actual = mo3.to_array()
    expected = ((1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, -1, 0),
                (0, 0, 0, -1))
    assert actual == expected

    mo4 = mo2.tensor(mo1)
    actual = mo4.to_array()
    expected = ((1, 0, 0, 0),
                (0, -1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, -1))
    assert actual == expected


def test_measurement_on_known_simple_states():
    """
    Deterministic if we know the state and measure along 
    the appropriate axis.
    """
    sv_0 = StateVector((1, 0))
    observable = pauli_z
    vals, states = observable.eig()
    state_index = -1
    sv_0_array = sv_0.to_array()
    for idx, state in enumerate(states):
        if np.array_equal(sv_0_array, state):
            state_index = idx
            break

    expected = vals[state_index]
    actual = observable.measure(sv_0)  # the state vector aligns with an eigenvector, so
    assert actual == expected


def test_new_state():
    """
    Hadamard:  measure once, confirm that the state updates 
    to correspond to the eigenstate of the eigenvalue
    """
    init_state = StateVector((1 / 2, 1 / 2))
    quantum_state = QuantumState(qubit_ids=[0])
    quantum_state.set_value(init_state)

    observable = pauli_z
    actual = quantum_state.measure(observable, on_qubits=[0])

    vals, states = observable.eig()
    if actual == 1:
        expected_state = (1, 0)
    elif actual == -1:
        expected_state = (0, 1)

    actual_state = states[vals.tolist().index(actual)]
    actual_state = tuple(actual_state.tolist())
    assert actual_state == expected_state


def test_simple_statistics():
    """
    Hadamard:  measure a bunch of independents, get 50/50
    """
    n_runs = 10000
    p_expected = 0.5
    mu = p_expected*n_runs
    sigma = math.sqrt(p_expected*p_expected*n_runs)

    observable = pauli_z
    vals, states = observable.eig()

    results = {}
    for v in vals:
        results[v] = 0

    for run in range(n_runs):
        init_state = StateVector((1/2, 1/2))
        actual = observable.measure(init_state)
        results[actual] += 1

    count1 = results[vals[0]]
    zscore = abs(count1 - mu)/sigma
    actual_rate = count1/n_runs

    # 3.7 Z-score is about 1 in 10K false-negative
    assert math.isclose(actual_rate, 0.5, abs_tol=3.7*zscore)


def test_second_measurement_consistency():
    """
    Polarized lenses check:  measure 0, then 90, get 0;
    This is equivalent to getting the same thing on the
    second measurement as on the first.
    """
    ev, es = pauli_z.eig()
    test_value = ev[0]
    good_samples = 0
    good_seconds = 0
    init_state = StateVector((1 / 2, 1 / 2))
    for n in range(1000):
        quantum_state = QuantumState(qubit_ids=[0])
        quantum_state.set_value(init_state)
        v = quantum_state.measure(pauli_z)
        if v != test_value:
            continue
        good_samples += 1
        v2 = quantum_state.measure(pauli_z)
        if v2 == test_value:
            good_seconds += 1
    assert good_seconds == good_samples  # 100% the same, nothing rotated


def test0_45_90():
    """
    measure 0, then 45, then 90, get about 1/8 transmission
    This should be ZXZ?
    """
    n_runs = 10000
    p_expected = 0.5
    mu = p_expected*n_runs
    sigma = math.sqrt(p_expected*p_expected*n_runs)

    evz, esz = pauli_z.eig()
    evx, esx = pauli_x.eig()

    good_count = 0
    for n in range(n_runs):
        a_system = StateVector((1, 1))
        v = pauli_z.measure(a_system)
        if v != evz[0]:  # only the first passes the filter
            continue
        v2 = pauli_x.measure(a_system)
        if v2 != evx[0]:  # only the first passes the filter
            continue
        v3 = pauli_z.measure(a_system)
        if v3 != evz[0]:  # only the first passes the filter
            continue

        good_count += 1

    actual_rate = good_count/n_runs
    zscore = abs(good_count - mu)/sigma
    assert math.isclose(actual_rate, 0.125, abs_tol=3.7*zscore)


def test_power_of_operator():
    expected = phase_op.to_array(as_numpy=True)
    actual = pauli_z ** (1/2)
    actual = actual.to_array(as_numpy=True)
    assert close_enough(actual, expected)


def test_2qubit_gate_on_bigger_state():
    # set up a 5-qubit state - every qubit defaults to |0>
    device = Device(qubits=5)
    # select a 2-qubit subsystem (#3 and #1)
    q1, q3 = device.get_qubits(keys=[1, 3])
    # initialize q3 to |1> to control the CNOT (X-gate is a classical NOT gate)
    PauliX(q3)
    # this should flip q1 from |0> to |1> because q3 = |1>
    CNOT(q3, q1)

    # this returns the eigenvalue
    actual = q3.measure(pauli_z)
    # map the measured value to |0> or |1>
    expected = 1

    assert actual == expected


def test_bell_state_coerce_density_matrix():
    device = Device(qubits=2)
    device.make_bell_pair([0, 1])

    qubit = device.get_qubit(0)
    state = qubit.get_state(force_density_matrix=True)

    expected_state = DensityMatrix(((0.5, 0, 0, 0.5),
                                    (0, 0, 0, 0),
                                    (0, 0, 0, 0),
                                    (0.5, 0, 0, 0.5)))

    actual_state_array = state.to_array(as_numpy=True)
    expected_state_array = expected_state.to_array(as_numpy=True)
    assert close_enough(actual_state_array, expected_state_array)


def test_bell_state_partial_trace():
    device = Device(qubits=2)
    device.make_bell_pair(qubits=[0, 1])

    qubit = device.get_qubit(0)
    state = qubit.get_state(force_density_matrix=True)

    #  this is the thing we partial trace on:  trace out Bob / qubit1
    alice_state = state.partial_trace(keep_qubits=[0])

    ket0 = StateVector((1, 0))
    ket1 = StateVector((0, 1))

    expected = (ket0.to_density_matrix() + ket1.to_density_matrix())
    expected = expected.scale(0.5)

    alice = alice_state.to_array(True).flat
    expected = expected.to_array(True).flat
    assert close_enough(alice, expected)


def test_partial_trace_00():
    ket = StateVector((1, 0, 0, 0))

    state = ket.to_density_matrix()
    reduced_state = state.partial_trace(keep_qubits=[0])
    reduced_state = reduced_state.to_array(as_numpy=True)

    single_ket = StateVector((1, 0))
    expected_state = single_ket.to_density_matrix().to_array(as_numpy=True)
    assert np.array_equal(expected_state, reduced_state)


def test_partial_trace_01():

    ket = StateVector((0, 1, 0, 0))

    state = ket.to_density_matrix()
    reduced_state = state.partial_trace(keep_qubits=[1])
    reduced_state = reduced_state.to_array(as_numpy=True)

    single_ket = StateVector((0, 1))
    expected_state = single_ket.to_density_matrix().to_array(as_numpy=True)
    assert np.array_equal(expected_state, reduced_state)


def test_partial_trace_01010():

    zero = StateVector.ZERO
    one = StateVector.ONE

    ket = zero.tensor(one)
    ket = ket.tensor(zero)
    ket = ket.tensor(one)
    ket = ket.tensor(zero)

    state = ket.to_density_matrix()
    reduced_state = state.partial_trace(keep_qubits=[3, 1])
    reduced_state = reduced_state.to_array(as_numpy=True)

    expected_state = one.tensor(one)
    expected_state = expected_state.to_density_matrix()
    expected_state = expected_state.to_array(as_numpy=True)

    assert np.array_equal(expected_state, reduced_state)
