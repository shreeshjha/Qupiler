#!/bin/bash
# run_benchmarks.sh - Quick start script for running benchmarks

echo "🚀 Quantum MLIR Benchmarking - Quick Start"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "ast_json_to_mlir.py" ]; then
    echo "❌ ast_json_to_mlir.py not found!"
    echo "   Please run this script from your project directory"
    exit 1
fi

# Check if benchmark script exists
if [ ! -f "quantum_mlir_benchmark.py" ]; then
    echo "❌ quantum_mlir_benchmark.py not found!"
    echo "   Please ensure the benchmark script is in this directory"
    exit 1
fi

# Create output directory
mkdir -p benchmark_results

echo "📁 Available C test files:"
find . -name "*.c" -type f | head -10

echo ""
echo "🔧 Starting benchmark analysis..."

# Method 1: Quick MLIR analysis (if you have existing MLIR files)
echo ""
echo "📊 Method 1: Quick MLIR Analysis"
echo "--------------------------------"
if ls *.mlir 1> /dev/null 2>&1; then
    echo "✅ Found existing MLIR files - analyzing..."
    python3 simple_metrics_analyzer.py --dir . --compare --output benchmark_results/quick_analysis.json
else
    echo "ℹ️  No existing MLIR files found"
fi

# Method 2: Full pipeline benchmark
echo ""
echo "🏃 Method 2: Full Pipeline Benchmark"
echo "-----------------------------------"

# Check dependencies
python3 -c "import pandas, matplotlib, seaborn, psutil, yaml" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Python dependencies available"
    
    # Run full benchmark if we have test files
    c_files=$(find . -name "*.c" -type f | wc -l)
    if [ $c_files -gt 0 ]; then
        echo "🚀 Running full benchmark on $c_files test files..."
        python3 quantum_mlir_benchmark.py --verbose
    else
        echo "⚠️  No C test files found for full benchmark"
    fi
else
    echo "⚠️  Missing Python dependencies - install with:"
    echo "   pip install pandas matplotlib seaborn psutil pyyaml"
    echo ""
    echo "🔄 Running basic analysis instead..."
    
    # Basic analysis without dependencies
    echo "Analyzing your pipeline manually..."
    
    # Try to run just the AST generation and analysis
    for c_file in *.c; do
        if [ -f "$c_file" ]; then
            echo "🔍 Analyzing: $c_file"
            
            # Generate AST
            clang -Xclang -ast-dump=json -fsyntax-only "$c_file" > "${c_file%.c}_ast.json" 2>/dev/null
            
            if [ $? -eq 0 ]; then
                echo "  ✅ AST generated"
                
                # Try to generate MLIR
                python3 ast_json_to_mlir.py "${c_file%.c}_ast.json" "${c_file%.c}.mlir" 2>/dev/null
                
                if [ $? -eq 0 ] && [ -f "${c_file%.c}.mlir" ]; then
                    echo "  ✅ MLIR generated"
                    
                    # Basic metrics
                    gates=$(grep -c "q\." "${c_file%.c}.mlir" 2>/dev/null || echo "0")
                    qubits=$(grep -o "%q[0-9]*" "${c_file%.c}.mlir" 2>/dev/null | sort -u | wc -l)
                    
                    echo "  📊 Basic metrics: $qubits qubits, $gates operations"
                else
                    echo "  ❌ MLIR generation failed"
                fi
            else
                echo "  ❌ AST generation failed"
            fi
            echo ""
        fi
    done
fi

# Method 3: Manual testing
echo ""
echo "🛠️  Method 3: Manual Testing Commands"
echo "------------------------------------"
echo "You can run individual components manually:"
echo ""
echo "1. Generate AST from C:"
echo "   clang -Xclang -ast-dump=json -fsyntax-only your_file.c > ast.json"
echo ""
echo "2. Generate MLIR from AST:"
echo "   python3 ast_json_to_mlir.py ast.json high_level.mlir"
echo ""
echo "3. Convert to gate-level:"
echo "   python3 gate_converter.py high_level.mlir gate_level.mlir"
echo ""
echo "4. Optimize:"
echo "   python3 gate_optimizer2.py gate_level.mlir optimized.mlir"
echo ""
echo "5. Analyze metrics:"
echo "   python3 simple_metrics_analyzer.py optimized.mlir"
echo ""

echo "📁 Results will be in: benchmark_results/"
echo ""
echo "✅ Benchmark analysis complete!"
echo ""
echo "📊 Next steps:"
echo "   • Check benchmark_results/ for detailed results"
echo "   • Run individual test cases manually if needed"
echo "   • Install missing dependencies for full benchmarking"
