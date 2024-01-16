# qoord
Tiny, rough-and-ready quantum circuit simulator for exploring quantum 
networking and computing.

Should only require Numpy to get started.

## Overview
Most quantum computing efforts are focused on "gate-based" quantum computers,
which are built from a network of quantum logic gates operating on _qubits_ 
(quantum bits).  These gates are quantum analogs of the logic gates (AND, OR, NOT) used in 
classical computers to manipulate the binary-valued bits.  Each
program is written as a sequence of quantum gates, which are applied to the qubits.
A quantum circuit simulator is a program to simulate the behaviour 
of a gate-based quantum computer.  Qoord is a quantum circuit simulator, written 
to teach myself about quantum computing, and (very secondarily) to prototype quantum 
algorithms.  

### Notes and caveats
Qoord is a very simple simulator, 
designed to be easy to understand and hack on.  It's not designed for speed or scale, 
which are challenging problems for a quantum simulator (because of the exponential 
growth in the size of the quantum state vector as the number of qubits increases.) 
Since quantum computation in Qoord involves repeated matrix multiplications, the 
accuracy will be limited by the standard floating point precision of Python and 
numpy - we currently don't take any measures to correct for this.


## Design
The base layer represents and manipulates quantum states using complex-number vectors.  
Each quantum state is a vector of length $2^n$, where $n$ is the number of qubits in the system.  To change a quantum state, 
we apply matrix operators through multiplication.  These operators are always either
_unitary_ (representing quantum gates) or _projections_ (representing measurements).
You need to apply measurements to extract information from a quantum state.

The core of the library is the `StateVector` class, which represents a 
quantum state as a vector of complex numbers.  The `StateVector` class 
is immutable, and all operations on it return a new `StateVector` instance.  
This makes it easy to reason about the state of a quantum system at any point 
in a circuit.


## Usage

### A quantum state is a vector of complex numbers.
You can create a quantum state by passing a list of complex numbers 
to the `StateVector` class.  Here's a state vector representing a
single qubit in the |0> state, and another in the |+> state.
```python
from qoord import StateVector
sv0 = StateVector([1, 0])  # |0>
sv_plus = StateVector([1, 1])  # |+>
```
State vectors are always normalised to have unit length.  
If you print out a state vector, you'll see the normalised version.
```python
print(sv_plus)
# StateVector([0.70710678+0.j, 0.70710678+0.j])
```

```

### Create a Bell pair
```


```