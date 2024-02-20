from qoord import Device, StateVector
from qoord.core_operators import pauli_z
from qoord.states import close_enough

"""
If Alice and Bob share only a source of entangled qubits, with no
other communication mechanism, they can use the qubits to generate 
matching random data.  They will each see a random sequence of 1s and -1s,
but their sequences will be exactly the same.  

Quantum Key Distribution uses this property to generate a shared secret
key that can be used without having to transmit the key over any channel.
They share qubits ahead of time, but the classical bit for the shared key 
does not exist until either Alice or Bob performs a local measurement.
"""


device = Device(qubits=2)
alice_results = []
bob_results = []

n_runs = 1000

standard_bell_state = StateVector((1, 0, 0, 1))
standard_check_result = []
for idx in range(n_runs):
    # re-set the device to the |00> state
    device.initialize(StateVector((1, 0, 0, 0)))

    # Alice and Bob share a Bell pair, a basic type of quantum entanglement
    pair = device.make_bell_pair(qubits=[0, 1])

    """ 
    We could be cheating, by generating identical random bits and
    sending them to Alice and Bob.  But we're not doing that!
    """

    # Check that the underlying state is the standard Bell state
    quantum_state = device.get_state()
    print(quantum_state)
    print(standard_bell_state)
    using_standard = close_enough(quantum_state, standard_bell_state)
    # default tolerance on this comparison is 1e-15
    standard_check_result.append(using_standard)

    """ 
    Since the generated state is the *same joint state* every time 
    before we give the qubits to Alice and Bob, we are not just
    generating two identical random bits ahead of time and sending
    them off.  The final bits are not determined until Alice and Bob
    perform their measurements.
    """
    alice_qubit = pair[0]  # Alice receives the first qubit
    bob_qubit = pair[1]  # Bob receives the second qubit

    # Alice measures her qubit in the standard basis
    result = alice_qubit.measure(pauli_z)
    alice_results.append(result)  # result is either -1 or 1

    # Bob measures his qubit in the standard basis
    result = bob_qubit.measure(pauli_z)
    bob_results.append(result)  # result is either -1 or 1

# All the trials should have used the standard Bell state
standard_check_pct = 100*sum(standard_check_result)/n_runs
print(f"Standard Bell state used in {standard_check_pct}% of trials.")

# Alice and Bob should have roughly 50% 1s and -1s
alice_pos = sum([1 for r in alice_results if r == 1])
bob_pos = sum([1 for r in bob_results if r == 1])
print(f"Alice: {100*alice_pos/n_runs}% value 1; {100*(1-alice_pos/n_runs)}% value -1")
print(f"Bob: {100*bob_pos/n_runs}% value 1; {100*(1-bob_pos/n_runs)}% value -1")

# The entanglement in the Bell pair ensures that Alice and Bob's
# results are perfectly correlated and they have the same sequence.
match = sum([1 for a, b in zip(alice_results, bob_results) if a == b])
print(f"Alice and Bob match on {100*match/n_runs}% of trials")


