#!/bin/bash
# setup_benchmark.sh - Setup script for Quantum MLIR Benchmarking

echo "🚀 Setting up Quantum MLIR Benchmarking System"
echo "=============================================="

# Check Python version
python3_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Python3 found: $python3_version"
else
    echo "❌ Python3 not found! Please install Python 3.7+"
    exit 1
fi

# Check if clang is available
clang_version=$(clang --version 2>&1 | head -n1)
if [[ $? -eq 0 ]]; then
    echo "✅ Clang found: $clang_version"
else
    echo "❌ Clang not found! Please install clang for AST generation"
    echo "   Ubuntu/Debian: sudo apt-get install clang"
    echo "   macOS: brew install llvm"
    exit 1
fi

# Create virtual environment
if [ ! -d "benchmark_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv benchmark_env
fi

# Activate virtual environment
source benchmark_env/bin/activate

# Install required packages
echo "📚 Installing Python dependencies..."
pip install --upgrade pip

# Core dependencies
pip install pandas matplotlib seaborn numpy psutil pyyaml

# Optional: Qiskit for circuit validation
echo "🔄 Installing Qiskit (optional)..."
pip install qiskit qiskit-aer

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Usage Instructions:"
echo "1. Activate the environment: source benchmark_env/bin/activate"
echo "2. Run benchmarks: python quantum_mlir_benchmark.py"
echo ""
echo "📁 Current test files found:"
find . -name "*.c" -type f | head -10

echo ""
echo "🎯 Quick start:"
echo "   python quantum_mlir_benchmark.py --verbose"
echo ""
echo "📖 For help:"
echo "   python quantum_mlir_benchmark.py --help"
