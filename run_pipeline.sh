#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 experiments/<file>.c"
  exit 1
fi

# 1) Grab and sanity-check input
C_PATH="$1"
if [[ ! -f "$C_PATH" ]]; then
  echo "Error: '$C_PATH' not found."
  exit 1
fi

# 2) Derive names
BASE="$(basename "$C_PATH" .c)"
EXP_DIR="$(dirname "$C_PATH")"

# 3) cd into experiments so all commands run there
pushd "$EXP_DIR" > /dev/null

echo "🔨 Building pipeline for $BASE.c in $(pwd)"
echo

# Step 1: C → AST JSON
echo "1) clang → AST JSON"
clang -Xclang -ast-dump=json -fsyntax-only \
      "$BASE.c" > "$BASE.json"

# Step 2: AST JSON → raw MLIR
echo "2) json_to_mlir"
python3 ../backend/ast_json_to_mlir.py \
      "$BASE.json" "$BASE.mlir"

# Step 3: MLIR → optimized MLIR
echo "3) mlir optimizer (--preserve-all)"
python3 ../backend/quantum_mlir_optimization_script.py \
      "$BASE.mlir" "$BASE"_opt.mlir \
      --preserve-all

# Step 4: opt MLIR → gate MLIR
echo "4) gate_converter"
python3 ../backend/gate_converter.py \
      "$BASE"_opt.mlir "$BASE"_gate.mlir

# Step 5: gate MLIR → optimized gate MLIR
echo "5) gate_optimizer3"
python3 ../backend/gate_optimizer2.py \
      "$BASE"_gate.mlir "$BASE"_gate_opt.mlir

# Step 6: gate-opt MLIR → circuit.py
echo "6) circuit_generator2"
python3 ../backend/circuit_generator2.py \
      "$BASE"_gate_opt.mlir circuit.py

echo
echo "✅ Done! Generated:"
ls -1 \
  "$BASE".{json,mlir}_opt*.mlir 2>/dev/null || true
echo " • $BASE.json"
echo " • $BASE.mlir"
echo " • $BASE_opt.mlir"
echo " • $BASE_gate.mlir"
echo " • $BASE_gate_opt.mlir"
echo " • circuit.py"

popd > /dev/null

