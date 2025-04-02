#!/usr/bin/env python3

import re
import sys

# Regular expression to match our MLIR instructions
FUNC_RE = re.compile(r'^func @(\w+)\(\) -> \(\) {')
ALLOC_RE = re.compile(r'^\s*%(\w+)\s*=\s*q\.alloc\s*:\s*!qreg<(\d+)>')
INIT_RE = re.compile(r'^\s*q\.init\s*%(\w+),\s*(\d+)\s*:\s*i32')
CX_RE = re.compile(r'^\s*q\.cx\s*%(\w+)\[(\d+)\],\s*%(\w+)\[(\d+)\]')
X_RE = re.compile(r'^\s*q\.x\s*%(\w+)\[(\d+)\]')
CCX_RE = re.compile(r'^\s*q\.ccx\s*%(\w+)\[(\d+)\],\s*%(\w+)\[(\d+)\],\s*%(\w+)\[(\d+)\]')
MEASURE_RE = re.compile(r'^\s*%(\w+)\s*=\s*q\.measure\s*%(\w+)\s*:\s*!qreg\s*->\s*i32')
PRINT_RE = re.compile(r'^\s*q\.print\s*%(\w+)')
CONST_RE = re.compile(r'^\s*%(\w+)\s*=\s*q\.const\s*(\d+)\s*:\s*i32')
RETURN_RE = re.compile(r'^\s*return')

# We'll collect registers definitions as we parse.
quantum_registers = {}
classical_registers = {}

# Helpers for our output code
def emit_header():
    header = [
        "from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister",
        "from qiskit_aer import AerSimulator",
        "from qiskit.visualization import plot_histogram",
        "# Helper to initialize a quantum register to a classical value",
        "def initialize_register(qc, qreg, value, num_bits):",
        "    # This is a placeholder: in practice, you need a circuit to load a binary number",
        "    # Here we assume the register is already in state |0> and then use X gates to set bits.",
        "",
        "    # IMPORTANT FIX: Reverse the bit string so qubit[0] is the LSB",
        "    bin_val = format(value, '0{}b'.format(num_bits))[::-1]",
        "    for i, bit in enumerate(bin_val):",
        "        if bit == '1':",
        "            qc.x(qreg[i])",
        "",
        "",
    ]

    return "\n".join(header)

def translate_line(line):
    # ALLOC
    m = ALLOC_RE.match(line)
    if m:
        reg, size = m.groups()
        quantum_registers[reg] = int(size)
        # Create quantum register code
        return f"{reg} = QuantumRegister({size}, '{reg}')"

    # INIT
    m = INIT_RE.match(line)
    if m:
        reg, value = m.groups()
        size = quantum_registers.get(reg, 0)
        return f"initialize_register(qc, {reg}, {value}, {size})"

    # CX
    m = CX_RE.match(line)
    if m:
        control, cidx, target, tidx = m.groups()
        return f"qc.cx({control}[{cidx}], {target}[{tidx}])"

    # X
    m = X_RE.match(line)
    if m:
        reg, idx = m.groups()
        return f"qc.x({reg}[{idx}])"

    # CCX
    m = CCX_RE.match(line)
    if m:
        c1, c1idx, c2, c2idx, target, tidx = m.groups()
        return f"qc.ccx({c1}[{c1idx}], {c2}[{c2idx}], {target}[{tidx}])"

    # MEASURE
    m = MEASURE_RE.match(line)
    if m:
        cl_reg, qreg = m.groups()
        # Create classical register for measurement result (if not exists)
        classical_registers[cl_reg] = quantum_registers.get(qreg, 0)
        return f"{cl_reg} = ClassicalRegister({quantum_registers.get(qreg, 0)}, '{cl_reg}')\n" \
               f"qc.measure({qreg}, {cl_reg})"

    # PRINT
    m = PRINT_RE.match(line)
    if m:
        reg = m.group(1)
        return f"print('Measurement result:', {reg})"

    # CONST (classical constant)
    m = CONST_RE.match(line)
    if m:
        reg, val = m.groups()
        classical_registers[reg] = 32  # Assume 32-bit, if needed
        return f"{reg} = {val}  # classical constant"

    # RETURN or blank
    m = RETURN_RE.match(line)
    if m:
        return ""

    # If nothing matched, ignore the line.
    return None

def translate_mlir(mlir_lines):
    output_lines = []
    func_name = "main"
    for line in mlir_lines:
        # Check for function header
        m = FUNC_RE.match(line)
        if m:
            func_name = m.group(1)
            output_lines.append(f"# Function: {func_name}")
            continue
        # Process each line
        trans = translate_line(line)
        if trans is not None:
            output_lines.append(trans)
    return output_lines

def main():
    if len(sys.argv) < 3:
        print("Usage: python qmlir_to_qiskit.py <input_mlir> <output_py>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r') as f:
        mlir_lines = f.readlines()

    translated_lines = translate_mlir(mlir_lines)

    # Start building Qiskit code
    qiskit_lines = []
    qiskit_lines.append(emit_header())
    # Declare registers: we have collected quantum_registers and classical_registers
    for reg, size in quantum_registers.items():
        qiskit_lines.append(f"{reg} = QuantumRegister({size}, '{reg}')")
    for reg, size in classical_registers.items():
        qiskit_lines.append(f"{reg} = ClassicalRegister({size}, '{reg}')")
    qiskit_lines.append("qc = QuantumCircuit(" +
                         ", ".join(list(quantum_registers.keys()) + list(classical_registers.keys())) + ")")
    qiskit_lines.append("")
    qiskit_lines.extend(translated_lines)
    qiskit_lines.append("")
    qiskit_lines.append("qc.draw(output='text')")
    qiskit_lines.append("qc.draw(output='mpl')  # if you want a matplotlib visualization")
    qiskit_lines.append("qc.decompose()")
    qiskit_lines.append("print(qc)")

    with open(output_file, 'w') as f:
        f.write("\n".join(qiskit_lines))
    print("Qiskit code generated successfully in", output_file)

if __name__ == '__main__':
    main()

