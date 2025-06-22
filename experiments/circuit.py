#!/usr/bin/env python3
'''
Generated Qiskit Circuit from Optimized MLIR

Expected classical result: 7
'''

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def convert_4bit_to_signed(unsigned_value):
    '''Convert 4-bit unsigned to signed interpretation'''
    if unsigned_value >= 8:
        return unsigned_value - 16
    else:
        return unsigned_value

def initialize_register(qc, qreg, value, num_bits):
    '''Initialize quantum register to classical value'''
    bin_val = format(value, f'0{num_bits}b')[::-1]  # LSB first
    for i, bit in enumerate(bin_val):
        if bit == '1':
            qc.x(qreg[i])
    print(f'   Initialized {qreg.name} to {value} (binary: {bin_val[::-1]})')

def apply_and_circuit(qc, a_reg, b_reg, result_reg):
    '''CORRECTED: Pure bitwise AND circuit using only necessary Toffoli gates'''
    print(f'   Applying pure AND circuit: {a_reg.name} & {b_reg.name} -> {result_reg.name}')
    # For bitwise AND: result[i] = a[i] & b[i] using Toffoli gates
    # Only add Toffoli gate if both input bits could be 1
    for i in range(min(len(a_reg), len(b_reg), len(result_reg))):
        qc.ccx(a_reg[i], b_reg[i], result_reg[i])
        print(f'     CCX: {a_reg.name}[{i}] & {b_reg.name}[{i}] -> {result_reg.name}[{i}]')

def apply_or_circuit(qc, a_reg, b_reg, result_reg):
    '''CORRECTED: Pure bitwise OR circuit'''
    print(f'   Applying pure OR circuit: {a_reg.name} | {b_reg.name} -> {result_reg.name}')
    # For bitwise OR: result[i] = a[i] | b[i]
    # Using: a | b = a ⊕ b ⊕ (a & b)
    for i in range(min(len(a_reg), len(b_reg), len(result_reg))):
        qc.cx(a_reg[i], result_reg[i])  # Copy a[i]
        qc.cx(b_reg[i], result_reg[i])  # XOR b[i]
        qc.ccx(a_reg[i], b_reg[i], result_reg[i])  # XOR (a[i] & b[i])
        print(f'     OR bit {i}: {a_reg.name}[{i}] | {b_reg.name}[{i}] -> {result_reg.name}[{i}]')

def apply_not_circuit(qc, input_reg, result_reg):
    '''CORRECTED: Pure bitwise NOT circuit'''
    print(f'   Applying pure NOT circuit: ~{input_reg.name} -> {result_reg.name}')
    # For bitwise NOT: result[i] = ~input[i]
    for i in range(min(len(input_reg), len(result_reg))):
        qc.cx(input_reg[i], result_reg[i])  # Copy input
        qc.x(result_reg[i])  # Flip bit
        print(f'     NOT bit {i}: ~{input_reg.name}[{i}] -> {result_reg.name}[{i}]')

def apply_xor_circuit(qc, a_reg, b_reg, result_reg):
    '''XOR circuit'''
    for i in range(min(len(a_reg), len(b_reg), len(result_reg))):
        qc.cx(a_reg[i], result_reg[i])
        qc.cx(b_reg[i], result_reg[i])

def apply_add_circuit(qc, a_reg, b_reg, result_reg):
    '''Addition circuit'''
    for i in range(min(len(a_reg), len(result_reg))):
        qc.cx(a_reg[i], result_reg[i])
    for i in range(min(len(b_reg), len(result_reg))):
        qc.cx(b_reg[i], result_reg[i])
    if len(result_reg) > 1 and len(a_reg) > 0 and len(b_reg) > 0:
        qc.ccx(a_reg[0], b_reg[0], result_reg[1])

def apply_sub_circuit(qc, a_reg, b_reg, result_reg):
    '''Subtraction circuit'''
    for i in range(min(len(a_reg), len(result_reg))):
        qc.cx(a_reg[i], result_reg[i])
    for i in range(min(len(b_reg), len(result_reg))):
        qc.cx(b_reg[i], result_reg[i])

def apply_mul_circuit(qc, a_reg, b_reg, result_reg):
    '''Multiplication circuit (simplified)'''
    for i in range(min(len(a_reg), len(result_reg))):
        if i < len(b_reg):
            qc.ccx(a_reg[i], b_reg[0], result_reg[i])

def apply_div_circuit(qc, a_reg, b_reg, result_reg):
    '''Division circuit (simplified)'''
    if len(a_reg) > 1 and len(result_reg) > 0:
        qc.cx(a_reg[1], result_reg[0])

def apply_mod_circuit(qc, a_reg, b_reg, result_reg):
    '''Modulo circuit'''
    for i in range(min(len(a_reg), len(result_reg))):
        qc.cx(a_reg[i], result_reg[i])
    if len(b_reg) > 0 and len(result_reg) > 0:
        qc.cx(b_reg[0], result_reg[0])

def apply_gt_circuit(qc, a_reg, b_reg, result_reg):
    '''Greater than comparison circuit: result = (a > b)'''
    # Simplified greater than comparison
    # For 4-bit numbers, we implement a basic comparison
    # This is a simplified version - real quantum comparison is more complex

    # Compare bit by bit from MSB to LSB
    # If a[i] = 1 and b[i] = 0 for any bit i, then a > b
    # We use ancilla qubits for intermediate results

    # Simple implementation: check if a[1] > b[1] (MSB comparison)
    # This gives a reasonable approximation for the comparison
    if len(a_reg) > 1 and len(b_reg) > 1 and len(result_reg) > 0:
        # Create temporary ancilla for NOT b[1]
        qc.x(b_reg[1])  # Flip b[1] temporarily
        qc.ccx(a_reg[1], b_reg[1], result_reg[0])  # a[1] AND NOT b[1]
        qc.x(b_reg[1])  # Restore b[1]
    else:
        # Fallback: simple bit copy for basic comparison
        if len(a_reg) > 0 and len(result_reg) > 0:
            qc.cx(a_reg[0], result_reg[0])

def apply_lt_circuit(qc, a_reg, b_reg, result_reg):
    '''Less than comparison circuit: result = (a < b)'''
    # a < b is equivalent to b > a
    apply_gt_circuit(qc, b_reg, a_reg, result_reg)

def apply_eq_circuit(qc, a_reg, b_reg, result_reg):
    '''Equality comparison circuit: result = (a == b)'''
    # Simple equality check for LSB
    if len(a_reg) > 0 and len(b_reg) > 0 and len(result_reg) > 0:
        qc.cx(a_reg[0], result_reg[0])
        qc.cx(b_reg[0], result_reg[0])
        qc.x(result_reg[0])  # Flip to get equality (both same = 0 XOR = 1)

def apply_ne_circuit(qc, a_reg, b_reg, result_reg):
    '''Not equal comparison circuit: result = (a != b)'''
    # XOR gives inequality
    if len(a_reg) > 0 and len(b_reg) > 0 and len(result_reg) > 0:
        qc.cx(a_reg[0], result_reg[0])
        qc.cx(b_reg[0], result_reg[0])

def apply_neg_circuit(qc, input_reg, result_reg):
    '''Negation circuit'''
    for i in range(min(len(input_reg), len(result_reg))):
        qc.cx(input_reg[i], result_reg[i])
        qc.x(result_reg[i])
    qc.x(result_reg[0])

def apply_post_inc_circuit(qc, input_reg, orig_reg, inc_reg):
    '''Post-increment circuit'''
    for i in range(min(len(input_reg), len(orig_reg))):
        qc.cx(input_reg[i], orig_reg[i])
    for i in range(min(len(input_reg), len(inc_reg))):
        qc.cx(input_reg[i], inc_reg[i])
    qc.x(inc_reg[0])

def apply_post_dec_circuit(qc, input_reg, orig_reg, dec_reg):
    '''Post-decrement circuit'''
    for i in range(min(len(input_reg), len(orig_reg))):
        qc.cx(input_reg[i], orig_reg[i])
    for i in range(min(len(input_reg), len(dec_reg))):
        qc.cx(input_reg[i], dec_reg[i])
    qc.x(dec_reg[0])

def create_quantum_circuit():
    '''Create the quantum circuit'''
    print('🔬 Creating quantum circuit...')

    q0 = QuantumRegister(4, 'q0')
    q1 = QuantumRegister(4, 'q1')
    q2 = QuantumRegister(4, 'q2')
    q3 = QuantumRegister(4, 'q3')
    q4 = QuantumRegister(4, 'q4')
    q5 = QuantumRegister(4, 'q5')
    c0 = ClassicalRegister(4, 'c0')
    qc = QuantumCircuit(q0, q1, q2, q3, q4, q5, c0)

    operations_log = []

    # Operation 1: Initialize %q0 = 9
    initialize_register(qc, q0, 9, 4)
    operations_log.append('Initialize q0 = 9')

    # Operation 2: Initialize %q1 = 2
    initialize_register(qc, q1, 2, 4)
    operations_log.append('Initialize q1 = 2')

    # Operation 3: X gate
    qc.x(q2[0])
    operations_log.append('X gate: q2[0]')

    # Operation 4: X gate
    qc.x(q2[0])
    operations_log.append('X gate: q2[0]')

    # Operation 5: X gate
    qc.x(q2[1])
    operations_log.append('X gate: q2[1]')

    # Operation 6: X gate
    qc.x(q2[1])
    operations_log.append('X gate: q2[1]')

    # Operation 7: X gate
    qc.x(q2[2])
    operations_log.append('X gate: q2[2]')

    # Operation 8: X gate
    qc.x(q2[2])
    operations_log.append('X gate: q2[2]')

    # Operation 9: X gate
    qc.x(q2[3])
    operations_log.append('X gate: q2[3]')

    # Operation 10: X gate
    qc.x(q2[3])
    operations_log.append('X gate: q2[3]')

    # Operation 11: CX gate
    qc.cx(q1[0], q3[0])
    operations_log.append('CNOT: q1[0] -> q3[0]')

    # Operation 12: X gate
    qc.x(q3[0])
    operations_log.append('X gate: q3[0]')

    # Operation 13: CX gate
    qc.cx(q1[1], q3[1])
    operations_log.append('CNOT: q1[1] -> q3[1]')

    # Operation 14: X gate
    qc.x(q3[1])
    operations_log.append('X gate: q3[1]')

    # Operation 15: CX gate
    qc.cx(q1[2], q3[2])
    operations_log.append('CNOT: q1[2] -> q3[2]')

    # Operation 16: X gate
    qc.x(q3[2])
    operations_log.append('X gate: q3[2]')

    # Operation 17: CX gate
    qc.cx(q1[3], q3[3])
    operations_log.append('CNOT: q1[3] -> q3[3]')

    # Operation 18: X gate
    qc.x(q3[3])
    operations_log.append('X gate: q3[3]')

    # Operation 19: X gate
    qc.x(q4[0])
    operations_log.append('X gate: q4[0]')

    # Operation 20: CX gate
    qc.cx(q0[0], q2[0])
    operations_log.append('CNOT: q0[0] -> q2[0]')

    # Operation 21: CX gate
    qc.cx(q3[0], q2[0])
    operations_log.append('CNOT: q3[0] -> q2[0]')

    # Operation 22: CX gate
    qc.cx(q4[0], q2[0])
    operations_log.append('CNOT: q4[0] -> q2[0]')

    # Operation 23: CCX gate
    qc.ccx(q0[0], q3[0], q4[1])
    operations_log.append('Toffoli: q0[0] & q3[0] -> q4[1]')

    # Operation 24: CCX gate
    qc.ccx(q0[0], q4[0], q4[1])
    operations_log.append('Toffoli: q0[0] & q4[0] -> q4[1]')

    # Operation 25: CCX gate
    qc.ccx(q3[0], q4[0], q4[1])
    operations_log.append('Toffoli: q3[0] & q4[0] -> q4[1]')

    # Operation 26: CX gate
    qc.cx(q0[1], q2[1])
    operations_log.append('CNOT: q0[1] -> q2[1]')

    # Operation 27: CX gate
    qc.cx(q3[1], q2[1])
    operations_log.append('CNOT: q3[1] -> q2[1]')

    # Operation 28: CX gate
    qc.cx(q4[1], q2[1])
    operations_log.append('CNOT: q4[1] -> q2[1]')

    # Operation 29: CCX gate
    qc.ccx(q0[1], q3[1], q4[2])
    operations_log.append('Toffoli: q0[1] & q3[1] -> q4[2]')

    # Operation 30: CCX gate
    qc.ccx(q0[1], q4[1], q4[2])
    operations_log.append('Toffoli: q0[1] & q4[1] -> q4[2]')

    # Operation 31: CCX gate
    qc.ccx(q3[1], q4[1], q4[2])
    operations_log.append('Toffoli: q3[1] & q4[1] -> q4[2]')

    # Operation 32: CX gate
    qc.cx(q0[2], q2[2])
    operations_log.append('CNOT: q0[2] -> q2[2]')

    # Operation 33: CX gate
    qc.cx(q3[2], q2[2])
    operations_log.append('CNOT: q3[2] -> q2[2]')

    # Operation 34: CX gate
    qc.cx(q4[2], q2[2])
    operations_log.append('CNOT: q4[2] -> q2[2]')

    # Operation 35: CCX gate
    qc.ccx(q0[2], q3[2], q4[3])
    operations_log.append('Toffoli: q0[2] & q3[2] -> q4[3]')

    # Operation 36: CCX gate
    qc.ccx(q0[2], q4[2], q4[3])
    operations_log.append('Toffoli: q0[2] & q4[2] -> q4[3]')

    # Operation 37: CCX gate
    qc.ccx(q3[2], q4[2], q4[3])
    operations_log.append('Toffoli: q3[2] & q4[2] -> q4[3]')

    # Operation 38: CX gate
    qc.cx(q0[3], q2[3])
    operations_log.append('CNOT: q0[3] -> q2[3]')

    # Operation 39: CX gate
    qc.cx(q3[3], q2[3])
    operations_log.append('CNOT: q3[3] -> q2[3]')

    # Operation 40: CX gate
    qc.cx(q4[3], q2[3])
    operations_log.append('CNOT: q4[3] -> q2[3]')

    # Operation 41: CCX gate
    qc.ccx(q0[3], q3[3], q5[0])
    operations_log.append('Toffoli: q0[3] & q3[3] -> q5[0]')

    # Operation 42: CCX gate
    qc.ccx(q0[3], q4[3], q5[0])
    operations_log.append('Toffoli: q0[3] & q4[3] -> q5[0]')

    # Operation 43: CCX gate
    qc.ccx(q3[3], q4[3], q5[0])
    operations_log.append('Toffoli: q3[3] & q4[3] -> q5[0]')

    # Operation 44: Measure %q2
    qc.measure(q2, c0)
    operations_log.append('Measure: q2 -> c0')

    return qc, operations_log

def run_simulation(qc, operations_log, expected_result=7):
    '''Run quantum simulation and analyze results'''
    print('🚀 Running quantum simulation...')
    print(f'Circuit: {qc.num_qubits} qubits, {qc.depth()} depth, {len(qc.data)} gates')
    print()
    
    # Show operations
    print('📋 Quantum Operations:')
    for i, op in enumerate(operations_log, 1):
        print(f'  {i:2d}. {op}')
    print()
    
    # Run simulation
    simulator = AerSimulator()
    job = simulator.run(qc, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    # Analyze results
    print('📊 Measurement Results:')
    sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    for bitstring, count in sorted_results[:10]:
        decimal = int(bitstring, 2)
        percentage = (count / 1024) * 100
        print(f'  {bitstring} (decimal: {decimal:2d}) -> {count:4d} shots ({percentage:5.1f}%)')
    
    most_frequent_bits, most_frequent_count = sorted_results[0]
    quantum_result_unsigned = int(most_frequent_bits, 2)
    quantum_result_signed = convert_4bit_to_signed(quantum_result_unsigned)
    print(f'🎯 Quantum Result (unsigned): {quantum_result_unsigned}')
    print(f'🎯 Quantum Result (signed): {quantum_result_signed}')
    
    print(f'🧮 Expected Result: {expected_result}')
    
    if quantum_result_unsigned == expected_result:
        print('   ✅ PERFECT MATCH!')
        accuracy = 'PERFECT'
    else:
        difference = abs(quantum_result_unsigned - expected_result)
        print(f'   ⚠️  Difference: {difference}')
        
        # Check if expected appears in results
        for bitstring, count in sorted_results:
            if int(bitstring, 2) == expected_result:
                percentage = (count / 1024) * 100
                print(f'   ✅ Expected result found with {percentage:.1f}% probability')
                break
        
        accuracy = f'DIFF_{difference}'
    
    return quantum_result_unsigned, accuracy, counts

def visualize_circuit(qc):
    '''Show circuit diagram'''
    print('\n📈 Circuit Visualization:')
    print(f'   Quantum qubits: {qc.num_qubits}')
    print(f'   Classical bits: {qc.num_clbits}')
    print(f'   Circuit depth: {qc.depth()}')
    print(f'   Total gates: {len(qc.data)}')
    
    # Gate count
    gate_counts = {}
    for instruction in qc.data:
        gate_name = instruction.operation.name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
    
    print('   Gate breakdown:')
    for gate, count in sorted(gate_counts.items()):
        print(f'     {gate}: {count}')
    
    # Circuit diagram
    try:
        print('\n   Circuit Diagram:')
        diagram = str(qc.draw(output='text', fold=-1))
        diagram_lines = diagram.split('\n')
        for line in diagram_lines[:30]:  # Show first 30 lines
            print(f'   {line}')
        if len(diagram_lines) > 30:
            print(f'   ... ({len(diagram_lines) - 30} more lines)')
    except Exception as e:
        print(f'   Circuit diagram error: {e}')

def main():
    '''Main execution function'''
    print('🚀 Quantum Circuit Execution')
    print('=' * 50)
    
    try:
        # Create circuit
        print('📦 Creating quantum circuit...')
        qc, operations_log = create_quantum_circuit()
        
        if not qc or not operations_log:
            print('❌ Failed to create circuit or operations log')
            return
        
        # Visualize
        visualize_circuit(qc)
        
        # Run simulation
        quantum_result, accuracy, all_counts = run_simulation(qc, operations_log, 7)
        
        # Final summary
        print('\n' + '=' * 50)
        print('🎊 Execution Complete!')
        print(f'   Quantum Result: {quantum_result}')
        print(f'   Expected Result: 7')
        print(f'   Accuracy: {accuracy}')
        
        return qc, operations_log, quantum_result
        
    except Exception as e:
        print(f'❌ Error during execution: {e}')
        import traceback
        traceback.print_exc()
        return None, None, None

# Execute immediately when script runs
print('🎬 STARTING QUANTUM CIRCUIT EXECUTION...')
print('='*60)
result = main()
if result and result[0]:
    print('\n✅ QUANTUM CIRCUIT EXECUTION COMPLETED SUCCESSFULLY!')
else:
    print('\n❌ QUANTUM CIRCUIT EXECUTION FAILED!')

# Also ensure it runs with python -c or import
if __name__ == '__main__':
    if 'result' not in locals():
        print('\n🔄 BACKUP EXECUTION...')
        backup_result = main()