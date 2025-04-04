# 🧬 QUPILER: Quantum Intermediate Representation Generator

QUPILER is a compiler backend that transforms simplified C-style Abstract Syntax Trees (ASTs) in JSON format into a custom gate-level Quantum MLIR (QMLIR). This tool enables generating low-level quantum circuits with support for classical arithmetic, qubit management, and quantum operations. The resulting QMLIR can then be translated into Qiskit-compatible Python code for simulation.

---

## 📁 Directory Structure

```plaintext
QUPILER
├── .vscode/                           # VSCode configuration
├── backend/
│   ├── direct_ir_gen.cpp              # Converts JSON AST → QMLIR (quantum-aware)
│   ├── ir_gen.cpp                     # Generic IR generation (classical)
│   ├── json_to_simplified_ast.cpp     # Frontend simplifier to convert ASTs
│   ├── json.hpp                       # nlohmann::json header
│   └── qmlir_ir.hpp                   # QMLIR structure and enums
├── dialect/
│   ├── dialect.cpp                    # Quantum logic emitters (adder, multiplier, etc.)
│   ├── dialect.hpp
│   ├── utils.cpp                      # Utility: temporary name generator
│   └── utils.hpp
├── experiments/
│   ├── test.c                         # C code example with quantum_circuit function
│   ├── other_experiment_files.c       # Other C files for testing
├── scripts/
│   ├── qmlir_to_qiskits.py            # Python converter: QMLIR → Qiskit-compatible code
│   └── qiskit.ipynb                   # Notebook for quantum circuit simulation
└── README.md
```

## 🧩 Program Structure & Workflow

QUPILER converts your C-style program into a quantum circuit simulation using the following steps:

---

### ✅ 1. Write Your C Code

Create a C file (e.g., `test.c`) where you define your quantum logic inside a function `void quantum_circuit()`, and call this function inside `int main()`:

```c
void quantum_circuit() {
    int a = 2;
    int b = 3;
    int c = a + b;  // Supported operations: addition, subtraction, multiplication (single digit only)
}

int main() {
    quantum_circuit();
    return 0;
}
```

### ✅ 2. Generate the JSON AST

Use Clang to produce a JSON-formatted AST from your C file:

```bash
clang -Xclang -ast-dump=json -fsyntax-only test.c > test.json
```

### ✅ 3. Build the IR Generator

Compile direct_ir_gen.cpp along with dialect.cpp and utils.cpp to create the IR generator:

```bash
clang++ -std=c++17 direct_ir_gen.cpp ../dialect/dialect.cpp ../dialect/utils.cpp -o direct_ir_gen
```

### ✅ 4. Generate the QMLIR File

Run the IR generator to convert test.json into test.mlir.

Without debug logs:
```bash
../backend/direct_ir_gen test.json test.mlir
```

With debug logs:
```bash
../backend/direct_ir_gen test.json test.mlir debug
```

### ✅ 5. Translate QMLIR to Qiskit Code

Use the provided Python script to generate Qiskit-compatible Python code from your QMLIR:

```bash
python3 qmlir_to_qiskits.py ../experiments/test.mlir test_qiskit.py
```

## 🔮 Roadmap & Future Enhancements

### ✅ Enhanced Compound Statement Support
Improve support for compound blocks to cover remaining edge cases.

### 🔧 Optimized T Gate Implementation
Implement T gates in a more optimized and hardware-aware fashion.

###🔁 Function Calling Support
Enable quantum functions to call other quantum functions.

###⚙️ LLVM and Custom Optimization
Add LLVM pass integration and implement custom QMLIR-level optimizations.
