# qoord
Tiny, rough-and-ready quantum circuit emulator for exploring quantum 
networking and computing.

Should only require [Numpy](https://numpy.org) to get started.  Tests in
`test__init__.py` can be run using the 
[`pytest`](https://docs.pytest.org/en/8.0.x/) framework from PyPy.  Python 3.10 or higher recommended.

## Overview
Qoord is a quantum circuit emulator, written to teach myself about quantum 
computing, and (secondarily) to prototype quantum algorithms.  

In ordinary computers, the deepest level of programming (directly on the 
chip) uses logic gates (AND, OR, NOT) to manipulate binary bits.  Most 
quantum computing efforts are also focused on this kind of "gate-based" 
computing - devices are built from quantum bits or _qubits_.  Quantum 
computers also use logical operations, but because qubits have a more 
complex behavior than binary logic, the quantum logic gates are very 
different.

Each quantum program is written as a sequence of quantum gates, which are 
applied to the qubits.  Because quantum programs are still low-level and 
operate directly on the hardware, they are often called _quantum 
circuits_.  Most of us do not have an ion trap or a near-absolute-zero
refrigerator hanging around to build quantum systems with, but we can mimic 
quantum computers in software:  a quantum circuit emulator is a program to 
mimic the behavior of an idealized gate-based quantum computer, as a 
substitute for having actual quantum hardware.  That's what Qoord does.

### Background Material
Getting comfortable with the bra-ket (<| and |>) notation makes a big 
difference; they're two types of vectors.  The kets |> are basically 
vector-valued data, and the bras <| are the coefficients of multivariable
linear functions.  The expression <y|x> is then just a dot product, 
e.g. applying the linear model to the data to get a scalar answer.  
In quantum circuits, the ket vectors are always the current state of 
the quantum register, and the various gate operations are multiplying 
the state by a matrix.  

There are a few stumbling blocks related to what kind of data a qubit value 
provides - a single qubit stores a 2-dimensional complex-valued vector of 
length 1.  That's way more data than a classical bit, but you can only 
extract one classical bit per measurement and all the rest of the qubit 
information is destroyed in the process.

I strongly recommend https://www.scottaaronson.com/qclec.pdf - Scott 
Aaronson's lecture notes for the first semester of quantum information 
theory.  The beginning parts are good for understanding what a qubit is 
and what a gate is doing.  Later he goes into algorithms and nonlocal
games, which are incredibly cool.  

There's also https://quantum.country, which
is introductory material in circuit-model quantum computing, by Michael Nielsen 
and Andy Matuschak.  Nielsen also co-wrote "Mike and Ike", the standard intro to 
quantum computing / quantum information textbook.  The Quantum Country 
site is part of his work on finding alternative ways to teach deep topics.

Generally, it helped me a lot to use multiple sets of lecture notes online 
to get several perspectives on the same material.  One point that gets
glossed over:  when considering a gate acting on a qubit, you often
need to know the matrix that operates on the entire state vector, not just
the one qubit that the gate is acting on.  When this happens, you are 
taking tensor products of matrices; the 2x2 identity matrix is used on the 
other qubits.  This probably isn't at all obvious if you're not used to 
working with tensor products and linear maps, especially coming from a 
software background.

### Notes and caveats
Qoord is a very simple emulator, designed to be easy to understand and 
hack on.  It's not designed for speed or scale, which are challenging 
problems for a quantum emulator because of the exponential growth in 
the size of the quantum state vector as the number of qubits 
increases.  Since quantum computation in Qoord involves repeated matrix 
multiplications, the accuracy will be limited by the standard floating point 
precision of Python and numpy - we currently don't take any measures to 
correct for this.

Many other circuit emulators are available, but I wrote my own to 
really lock in the basics of the theory.  I've also used Cirq from Google,
IBM's Qiskit, and PennyLane from Xanadu.  They're all great, and they have
different strengths.  When I started this project in early 2023, one missing
feature from a lot of the packages out there was an ability to perform mid-circuit
measurements and then continue the circuit - because most of these were written to
control actual hardware, and most of the time you just measure at the end. 
I made sure Qoord could do mid-circuit measurements because I needed it for something 
else.  Good news - in the last year, most of the packages I look at have added this feature.

#### Pronunciation
Qoord is pronounced like "coordinate".  If you say it like "cord" or 
"qword", I probably won't notice.   My whole family are writers,
but I've mostly fought off the temptation to spell it "qoörd" with a 
[diaeresis](https://www.newyorker.com/culture/culture-desk/the-curse-of-the-diaeresis).  Although...
the two dots do look a bit entangled, so maybe I'll rethink it.


## Design
### Core: State Vectors and Matrix Operators
The base layer represents and manipulates program states using vectors of 
complex numbers, in the `StateVector` class.  Each state is a complex-valued
vector of length $2^n$, where $n$ is the number of qubits in the 
system.  This layer is primarily implementing mathematical operations.  It 
doesn't know anything about the physical interpretation as quantum states 
and gates.

The `StateVector` class is immutable, and all operations on it return a 
new `StateVector` instance.  States can also be represented as a 
`DensityMatrix`, an array of complex numbers that captures a broader set 
of possibilities where the quantum state is only partially determined.  All
`StateVector` instances can be converted to valid `DensityMatrix` 
instances, but not vice-versa.

To change a program's state, we multiply the `StateVector` by a matrix 
operator to get a new `StateVector`.  These matrix operators are always 
either _unitary_ (representing quantum gates) or _projections_ (representing 
measurements).  You use unitary matrices to change the program state during
a calculation; you use projection matrices to extract data from the program 
by reading the value of a qubit.  Operators are represented by the 
`MatrixOperator` class.

### Mantle: Quantum States and Gates

The `QuantumState` class represents the joint state of a set of 
qubits.  `QuantumState` contains a collection of qubit identifiers and 
either a `StateVector` or a `DensityMatrix` instance to represent the 
numeric values of the state.

When working with multiple qubits, the global state of the system can't be
broken down into a simple combination of the individual qubit states.  If
Alice and Bob's qubits are _entangled_, when Alice manipulates her qubit,
the global state of the system changes in a way that matters for Bob's qubit,
even if they are separated by a large distance and can't otherwise
interact.  This means that multiple distinct objects need to keep references 
to the global `QuantumState`; this violates the normal object/state 
encapsulation you want in software, but is a critical part of the quantum 
behavior.  We handle this by making all _references_ to the global 
`QuantumState` immutable, but the `QuantumState` itself is a mutable object
whose value is maintained by either a `StateVector` or a `DensityMatrix`.  All 
changes in the system involve updating the internal values of the shared
`QuantumState` object.

When constructing a quantum system, we first fix the number of qubits $n$ 
and initialize a `StateVector` to the ${\left|0\right\rangle}^n$ state.  The
`StateVector` is used to set up a `QuantumState` instance.  Then we 
create $n$ `Qubit` instances, passing the `QuantumState` to each constructor 
so the state is shared by all the qubits.  This reference is immutable, so 
qubits cannot lose their connection to the global state object.  However, 
because the `QuantumState` class has mutable internal state, gates and
measurements on a `Qubit` can change the global state of the system, 
and all the qubits still share access to the changed state.

| Object | Description |
|---|---|
| StateVector, DensityMatrix  | immutable wrapper around a numpy array  or numeric list.  Fundamental operations are just math. |
| QuantumState | mutable container for a state vector or density matrix, with a list of associated qubit identifiers. |
|Qubit, QubitSet | immutable identifier for one or more of the qubits in a quantum system, with an immutable reference to the QuantumState object. |



### Crust: Devices and Circuits
Users typically will initialize a `Device` instance with some number of
quantum bits.  The Device is a container and initializer for the shared 
`QuantumState`.

```python
    device = Device(qubits=2)
    device.initialize(StateVector((0, 0, 1, 0)))  # |10>

    qubits = device.get_qubits([0, 1])
    CNOT(qubits)

    expected = StateVector((0, 0, 0, 1))  # |11>
    actual = device.get_state()
    
    print(expected)
    print(actual)

    # (0, 0, 0, 1)
    # (0, 0, 0, 1)
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


### Create a Bell pair on two qubits
```python
device = Device(qubits=2)
device.make_bell_pair(qubits=[0, 1])

qubit = device.get_qubit(0)
state = qubit.get_state(force_density_matrix=True)

#  Use partial trace to reduce to look at just the first qubit
qb0_state = state.partial_trace(keep_qubits=[0])
print(qb0_state)  
# this is a density matrix, not a state vector,because of the 
# partial trace operation

```
