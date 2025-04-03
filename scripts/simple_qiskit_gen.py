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
inv0 = QuantumRegister(4, 'inv0')
one0 = QuantumRegister(4, 'one0')
b20 = QuantumRegister(4, 'b20')
c0 = QuantumRegister(5, 'c0')
c1 = QuantumRegister(5, 'c1')
t0 = ClassicalRegister(4, 't0')
qc = QuantumCircuit(q0, q1, q2, q3, inv0, one0, b20, c0, c1, t0)

# Function: main
initialize_register(qc, q0, 3, 4)
initialize_register(qc, q1, 1, 4)
qc.cx(q1[0], inv0[0])
qc.x(inv0[0])
qc.cx(q1[1], inv0[1])
qc.x(inv0[1])
qc.cx(q1[2], inv0[2])
qc.x(inv0[2])
qc.cx(q1[3], inv0[3])
qc.x(inv0[3])
initialize_register(qc, one0, 1, 4)
qc.cx(inv0[0], b20[0])
qc.cx(one0[0], b20[0])
qc.ccx(inv0[0], one0[0], c0[1])
qc.cx(inv0[1], b20[1])
qc.cx(one0[1], b20[1])
qc.ccx(inv0[1], one0[1], c0[2])
qc.ccx(b20[1], c0[1], c0[2])
qc.cx(c0[1], b20[1])
qc.cx(inv0[2], b20[2])
qc.cx(one0[2], b20[2])
qc.ccx(inv0[2], one0[2], c0[3])
qc.ccx(b20[2], c0[2], c0[3])
qc.cx(c0[2], b20[2])
qc.cx(inv0[3], b20[3])
qc.cx(one0[3], b20[3])
qc.ccx(inv0[3], one0[3], c0[4])
qc.ccx(b20[3], c0[3], c0[4])
qc.cx(c0[3], b20[3])
qc.cx(q0[0], q3[0])
qc.cx(b20[0], q3[0])
qc.ccx(q0[0], b20[0], c1[1])
qc.cx(q0[1], q3[1])
qc.cx(b20[1], q3[1])
qc.ccx(q0[1], b20[1], c1[2])
qc.ccx(q3[1], c1[1], c1[2])
qc.cx(c1[1], q3[1])
qc.cx(q0[2], q3[2])
qc.cx(b20[2], q3[2])
qc.ccx(q0[2], b20[2], c1[3])
qc.ccx(q3[2], c1[2], c1[3])
qc.cx(c1[2], q3[2])
qc.cx(q0[3], q3[3])
qc.cx(b20[3], q3[3])
qc.ccx(q0[3], b20[3], c1[4])
qc.ccx(q3[3], c1[3], c1[4])
qc.cx(c1[3], q3[3])
qc.measure(q3, t0)
print('Measurement result:', t0)


# Use the automatic simulator to choose the best method
simulator = AerSimulator(method='matrix_product_state')
job = simulator.run(qc, shots=1024)
result = job.result()
counts = result.get_counts()
print('\nMeasurement results:', counts)

# Uncomment to see circuit visualization
# qc.draw(output='text')
# qc.draw(output='mpl')  # if you want a matplotlib visualization

# Convert the most frequent result to decimal
if len(counts) > 0:
    most_frequent_result = max(counts, key=counts.get)
    decimal_result = int(most_frequent_result, 2)
    print(f"Circuit output: {decimal_result} (binary: {most_frequent_result})")