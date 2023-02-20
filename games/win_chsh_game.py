import random
from qoord import Hadamard as H, CNOT, \
    PauliX as X, PauliZ as Z, Device, binary_from_measurement


def score_game(alice_input, bob_input, alice_output, bob_output):
    return alice_input & bob_input == alice_output ^ bob_output


n_runs = 1000
results = []
final_scores = []
for _ in range(n_runs):
    # Alice and Bob entangle two qubits and each takes one
    device = Device(qubits=4)

    alice_q = device.get_qubits([0, 1])
    as0, al0 = alice_q

    bob_q = device.get_qubits([2, 3])
    bs0, bl0 = bob_q

    # create a Bell pair from the shared qubits of each side
    H(as0)
    CNOT(as0, bs0)

    # orient Alice to different measurement basis
    (X**-0.25)(as0)

    # then Alice and Bob each get an input of 0 or 1
    ref_input_alice = random.randint(0, 1)
    ref_input_bob = random.randint(0, 1)

    # set the local qubits to the value of the ref input
    if ref_input_alice == 1:
        # set it to |1>
        X(al0)

    if ref_input_bob == 1:
        X(bl0)

    # they each combine their inputs with their entangled qubits
    (CNOT ** 0.5)(al0, as0)  # Alice
    (CNOT ** 0.5)(bl0, bs0)  # Bob

    bob_observable = Z
    bob_does = binary_from_measurement(bob_observable, bs0)

    alice_observable = Z
    alice_does = binary_from_measurement(alice_observable, as0)

    s = score_game(ref_input_alice, ref_input_bob, alice_does, bob_does)

    result = (ref_input_alice, ref_input_bob, alice_does, bob_does, s)
    results.append(result)
    final_scores.append(s)

print(final_scores)
print(sum(final_scores))
