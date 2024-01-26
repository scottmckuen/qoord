# qoord
Tiny, rough-and-ready quantum circuit simulator for exploring quantum 
networking and computing.

Should only require Numpy to get started.

## Overview
Qoord is a quantum circuit simulator, written to teach myself about quantum 
computing, and (very secondarily) to prototype quantum algorithms.  

In ordinary computers, the deepest level of programming (directly on the chip) 
uses logic gates (AND, OR, NOT) to manipulate binary bits.  Most quantum 
computing efforts are also focused on this kind of "gate-based" computing -
devices are built from quantum bits or _qubits_.  Quantum computers also use 
logical operations, but because qubits have a more complex behavior than binary 
logic, the quantum logic gates are very different.

Each quantum program is written as a sequence of quantum gates, which are applied 
to the qubits.  Because quantum programs are still low-level and operate directly on
the hardware, they are often called _quantum circuits_.  A quantum circuit simulator 
is a program to simulate the behaviour of a gate-based quantum computer, as a substitute
for having actual hardware.  

### Notes and caveats
Qoord is a very simple simulator, designed to be easy to understand and 
hack on.  It's not designed for speed or scale, which are challenging 
problems for a quantum simulator (because of the exponential growth in 
the size of the quantum state vector as the number of qubits increases.)  
Since quantum computation in Qoord involves repeated matrix multiplications, 
the accuracy will be limited by the standard floating point precision of 
Python and numpy - we currently don't take any measures to correct for this.


## Design
### Core: states and operators
The base layer represents and manipulates program states using vectors of 
complex numbers, in the `StateVector` class.  Each state is a complex-valued
vector of length $2^n$, where $n$ is the number of qubits in the system.  
This layer is primarily implementing mathematical operations.  It doesn't 
know anything about the physical interpretation as quantum states and gates.

The `StateVector` class is immutable, and all operations on it return a 
new `StateVector` instance.  States can also be represented as a 
`DensityMatrix`, an array of complex numbers that captures a broader set 
of possibilities where the quantum state is only partially determined.
All `StateVector` instances can be converted to valid `DensityMatrix` 
instances, but not vice-versa.

To change a program's state, we multiply the `StateVector` by a matrix 
operator to get a new `StateVector`.  These matrix operators are always 
either _unitary_ (representing quantum gates) or _projections_ (representing 
measurements).  You use unitary matrices to change the program state during
a calculation; you use projection matrices to extract data from the program 
by reading the value of a qubit.  Operators are represented by the 
`MatrixOperator` class.

### Quantum states and gates

The `QuantumState` class represents the joint quantum state of a collection
of qubits.  `QuantumState` contains a collection of qubit identifiers and 
either a `StateVector` or a `DensityMatrix` instance to represent the numeric
values of the state.

When working with multiple qubits, the global state of the system can't be
broken down into a simple combination of the individual qubit states.  If
Alice and Bob's qubits are _entangled_, when Alice acts on her qubit,
the global state of the system changes in a way that matters for Bob's qubit.
This means that the references to the state are shared by multiple objects,
which violates the normal object/state encapsulation you want in software,
but is a critical part of the quantum behavior.  Here's how we handle it:

When constructing a quantum system, we first fix the number of qubits $n$ 
and initialize a `StateVector` to the ${\left|0\right\rangle}^n$ state.  The
`StateVector`is used to set up a `QuantumState` instance.  Then we 
create $n$ `Qubit` instances, passing the `QuantumState` to each constructor 
so the state is shared by all the qubits.  This reference is immutable, so 
qubits cannot lose their connection to the global state object.  However, 
because the `QuantumState` class has mutable internal state, gates and
measurements on a `Qubit` can change the global state of the system, 
and all the qubits still share access to the changed state.

``` 
StateVector | DensityMatrix  - immutable wrapper around a numpy array 
                or numeric list.  Fundamental operations are just math.

QuantumState - mutable container for a state vector or density matrix, 
               with a list of associated qubit identifiers.

Qubit, QubitSet - immutable identifier for one or more of the qubits in 
                a quantum system, with an immutable reference to the
                QuantumState object.

```


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