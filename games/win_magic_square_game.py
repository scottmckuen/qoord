import itertools
from itertools import product
import random
import cmath

from qoord import Identity as I, PauliX as X, PauliY as Y, PauliZ as Z, Device

strategy = [[I.tensor(Z), Z.tensor(I), Z.tensor(Z)],
            [X.tensor(I), I.tensor(X), X.tensor(X)],
            [X.tensor(Z), Z.tensor(X), Y.tensor(Y)]]

n_trials = 100
arange = range(3)
brange = range(3)
for alice_input, bob_input in itertools.product(arange, brange):
    #print(f"{alice_input}, {bob_input}")
    alice_row = strategy[alice_input]
    bob_col = [r[bob_input] for r in strategy]
    #print(alice_row)
    #print(alice_row[bob_input])
    #print(bob_col)
    #print(bob_col[alice_input])
    scores = []
    for _ in range(n_trials):
        #alice_input = random.randint(0, 2)
        #bob_input = random.randint(0, 2)

        d = Device(qubits=4)

        pair1 = d.make_bell_pair([0, 1])
        #print(pair1._global_state)
        pair2 = d.make_bell_pair([2, 3])
        #print(pair1._global_state)
        #print(pair2._global_state)
        #exit()

        alice_qubits = d.get_qubits([0, 2])
        bob_qubits = d.get_qubits([1, 3])

        #print(f"Alice: {alice_input}; Bob: {bob_input}")

        alice_results = []
        for gate in alice_row:
            op = gate.to_operator()
            value = alice_qubits.measure(op)
            alice_results.append(value)
        #print(alice_results[bob_input])

        bob_results = []
        for gate in bob_col:
            op = gate.to_operator()
            value = bob_qubits.measure(op)
            bob_results.append(value)
        #print(bob_results[alice_input])

        score = cmath.isclose(alice_results[bob_input], bob_results[alice_input])
        #print(score)
        scores.append(score)

    #print(scores)
    print(sum(scores))

sys.exit()


