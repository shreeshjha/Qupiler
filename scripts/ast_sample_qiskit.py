from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
# Helper to initialize a quantum register to a classical value
def initialize_register(qc, qreg, value, num_bits):
    # This is a placeholder: in practice, you need a circuit to load a binary number
    # Here we assume the register is already in state |0> and then use X gates to set bits.

    # IMPORTANT FIX: Reverse the bit string so qubit[0] is the LSB
    bin_val = format(value, '0{}b'.format(num_bits))[::-1]
    for i, bit in enumerate(bin_val):
        if bit == '1':
            qc.x(qreg[i])


q0 = QuantumRegister(4, 'q0')
q1 = QuantumRegister(4, 'q1')
q2 = QuantumRegister(4, 'q2')
q3 = QuantumRegister(4, 'q3')
c0 = QuantumRegister(5, 'c0')
t0 = ClassicalRegister(4, 't0')
qc = QuantumCircuit(q0, q1, q2, q3, c0, t0)

# Function: main
q0 = QuantumRegister(4, 'q0')
initialize_register(qc, q0, 3, 4)
q1 = QuantumRegister(4, 'q1')
initialize_register(qc, q1, 2, 4)
q2 = QuantumRegister(4, 'q2')
q3 = QuantumRegister(4, 'q3')
c0 = QuantumRegister(5, 'c0')
qc.cx(q0[0], q3[0])
qc.cx(q1[0], q3[0])
qc.ccx(q0[0], q1[0], c0[1])
qc.cx(q0[1], q3[1])
qc.cx(q1[1], q3[1])
qc.ccx(q0[1], q1[1], c0[2])
qc.ccx(q3[1], c0[1], c0[2])
qc.cx(c0[1], q3[1])
qc.cx(q0[2], q3[2])
qc.cx(q1[2], q3[2])
qc.ccx(q0[2], q1[2], c0[3])
qc.ccx(q3[2], c0[2], c0[3])
qc.cx(c0[2], q3[2])
qc.cx(q0[3], q3[3])
qc.cx(q1[3], q3[3])
qc.ccx(q0[3], q1[3], c0[4])
qc.ccx(q3[3], c0[3], c0[4])
qc.cx(c0[3], q3[3])
t0 = ClassicalRegister(4, 't0')
qc.measure(q3, t0)
print('Measurement result:', t0)


qc.draw(output='text')
qc.draw(output='mpl')  # if you want a matplotlib visualization
qc.decompose()
print(qc)