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
python3 ../backend/core/ast_json_to_mlir.py \
      "$BASE.json" "$BASE.mlir"

# NEW STEP 2.5: Extract expected result from high-level MLIR
echo "2.5) extract_expected_result"
python3 ../backend/generators/extract_expected_result.py \
      "$BASE.mlir" expected_res.txt

# Show the extracted expected result
if [[ -f expected_res.txt ]]; then
    EXPECTED_RESULT=$(cat expected_res.txt)
    echo "    📊 Expected result extracted: $EXPECTED_RESULT"
else
    echo "    ⚠️  Warning: expected_res.txt not created"
fi

# Step 3: MLIR → optimized MLIR
echo "3) mlir optimizer (--preserve-all)"
python3 ../backend/optimizers/quantum_mlir_optimization_script.py \
      "$BASE.mlir" "$BASE"_opt.mlir \
      

# Step 4: opt MLIR → gate MLIR
echo "4) gate_converter"
python3 ../backend/core/gate_converter.py \
      "$BASE"_opt.mlir "$BASE"_gate.mlir

# Step 5: gate MLIR → optimized gate MLIR
echo "5) gate_optimizer"
python3 ../backend/optimizers/gate_optimizer.py \
      "$BASE"_gate.mlir "$BASE"_gate_opt.mlir

# NEW Step 5.5: Enhanced quantum gate optimizations
echo "5.5) enhanced_optimizer (new quantum optimizations)"
python3 ../backend/optimizers/optimizer.py \
      "$BASE"_gate_opt.mlir "$BASE"_enhanced_opt.mlir

# MODIFIED Step 6: enhanced-opt MLIR → circuit.py (now uses expected_res.txt)
echo "6) circuit_generator (with correct expected result)"
python3 ../backend/generators/circuit_generator2.py \
      "$BASE"_enhanced_opt.mlir circuit.py expected_res.txt

echo
echo "✅ Done! Generated:"
echo " • $BASE.json"
echo " • $BASE.mlir"
echo " • $BASE_opt.mlir"
echo " • $BASE_gate.mlir"
echo " • $BASE_gate_opt.mlir"
echo " • $BASE_enhanced_opt.mlir"
echo " • expected_res.txt (expected result: $(cat expected_res.txt 2>/dev/null || echo 'N/A'))"
echo " • circuit.py"

echo
echo "🎯 Next steps:"
echo " • Run the circuit: python3 circuit.py"
echo " • Verify quantum result matches expected result: $(cat expected_res.txt 2>/dev/null || echo 'N/A')"

# Optional: Clean up expected_res.txt after pipeline
read -p "🗑️  Remove expected_res.txt? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f expected_res.txt
    echo "   🗑️  Removed expected_res.txt"
fi

popd > /dev/null
