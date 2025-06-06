#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <filesystem>
#include "backend/passes/quantum_fusion_pass.h"

namespace fs = std::filesystem;

/*
static void die(const char* msg) {
    std::cerr << "Error: " << msg << "\n";
    std::exit(1);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./main <source_file.c> [--run-tests]\n";
        return 1;
    }
    
    // Check if we're running the test suite 
    bool run_tests = false;
    for (int i = 0; i < argc; i++) {
        if(std::string(argv[i]) == "--run-tests") {
            run_tests = true;
            std::cout << "Running optimization test suite...\n";
            // Code to run the test suite would go here
            // For now, just compile and run the test program separately
            std::string test_cmd = "g++ -std=c++17 optimization_test.cpp -o run_tests && ./run_tests";
            std::system(test_cmd.c_str());
            return 0;
        }
    }
    std::string c_file = argv[1];
    if (!fs::exists(c_file)) {
        die(("File " +  c_file + " does not exist.").c_str());
        return 1;
    }

// Get the directory where our executable is located
    fs::path exe_path = fs::path(argv[0]);
    fs::path exe_dir = exe_path.parent_path();
    
    // Build paths relative to executable location
    fs::path backend_dir = exe_dir.parent_path() / "backend";
    fs::path scripts_dir = exe_dir.parent_path() / "scripts";
    
    std::string base = fs::path(c_file).stem();
    std::string json_file = base + ".json";
    std::string mlir_file = base + ".mlir";
    std::string opt_mlir  = base + ".opt.mlir";
    std::string qiskit_file = base + "_qiskit.py";

    // 1) Generating AST JSON
    std::string cmd_ast = "clang -Xclang -ast-dump=json -fsyntax-only " + c_file + " > " + json_file;
    std::cout << "Running: " << cmd_ast << "\n";
    if (std::system(cmd_ast.c_str()) != 0) {
        die("AST dump failed");
    }
    
    // 2) Generate raw MLIR
    std::string ir_exec = (backend_dir / "direct_ir_gen").string();
    if (!fs::exists(ir_exec)) {
        std::string cmd_compile = "clang++ -std=c++17 " + 
                                (backend_dir / "direct_ir_gen.cpp").string() + " " +
                                (exe_dir.parent_path() / "dialect/dialect.cpp").string() + " " +
                                (exe_dir.parent_path() / "dialect/utils.cpp").string() + 
                                " -o " + ir_exec;
        std::cout << "Compiling IR generator...\n";
        if (std::system(cmd_compile.c_str()) != 0) {
            die("Failed to compile IR generator.");
        }
    }

    std::string cmd_ir = ir_exec + " " + json_file + " " + mlir_file;
    std::cout << "Running: " << cmd_ir << "\n";
    if (std::system(cmd_ir.c_str()) != 0) {
        die("IR generation failed.");
    }
    
    // 3) Optimize MLIR (quantum-safe passes)
    // Direct optimization using our integrated function instead of mlir-opt
    std::cout << "Optimizing MLIR with integrated passes...\n";
    if (!optimizeMlirFile(mlir_file, opt_mlir)) {
        die("MLIR optimization failed");
    }
    
    // 4) Convert MLIR to Qiskit Python
    std::string cmd_qiskit = "python3 " + (scripts_dir / "qmlir_to_qiskits.py").string() + " " + mlir_file + " " + qiskit_file;
    std::cout << "Running: " << cmd_qiskit << "\n";
    if (std::system(cmd_qiskit.c_str()) != 0) {
        die("Qiskit translation failed");
    }

    // 5) Convert optimized MLIR to Qiskit Python 
    std::string cmd_qiskit_opt = "python3 " + (scripts_dir / "qmlir_to_qiskits.py").string() + " " + opt_mlir + " " + qiskit_file;
    std::cout << "Running: " << cmd_qiskit_opt << "\n";
    if (std::system(cmd_qiskit_opt.c_str()) != 0) {
        die("Qiskit translation failed.");
    }

    std::cout << "✅ Qiskit code generated: " << qiskit_file << "\n";
    std::cout << "📄 --- Qiskit Code ---\n";

    std::ifstream qiskit_in(qiskit_file);
    if (qiskit_in) {
        std::string line;
        while (std::getline(qiskit_in, line)) {
            std::cout << line << '\n';
        }
    } else {
        die("Failed to open generated Qiskit file.");
    }

    return 0;
}
*/ 

static void die(const char* msg) {
    std::cerr << "Error: " << msg << "\n";
    std::exit(1);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./main <source_file.c> [--run-tests]\n";
        return 1;
    }

    // Check if we're running the test suite
    bool run_tests = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--run-tests") {
            run_tests = true;
            std::cout << "Running optimization test suite...\n";
            // Find the optimization_test executab;e in the same directory as frontend_driver
            fs::path executable_path = fs::path(argv[0]).parent_path() / "optimization_test";
    
    std::cout << "Looking for optimization_test at: " << executable_path << "\n";
    
    if (fs::exists(executable_path)) {
        std::cout << "Found optimization_test executable, running tests...\n";
        return std::system(executable_path.string().c_str());
    } else {
        std::cerr << "Error: optimization_test executable not found at " << executable_path << "\n";
        std::cerr << "Make sure to build it with 'cmake .. && make'\n";
        return 1;
        }
            
        }
    }

    std::string c_file = argv[1];
    if (!fs::exists(c_file)) {
        die(("File " +  c_file + " does not exist.").c_str());
        return 1;
    }

    // Get the directory where our executable is located
    fs::path exe_path = fs::path(argv[0]);
    fs::path exe_dir = exe_path.parent_path();
    
    // Build paths relative to executable location
    fs::path backend_dir = exe_dir.parent_path() / "backend";
    fs::path scripts_dir = exe_dir.parent_path() / "scripts";
    
    std::string base = fs::path(c_file).stem();
    std::string json_file = base + ".json";
    std::string mlir_file = base + ".mlir";
    std::string opt_mlir  = base + ".opt.mlir";
    std::string qiskit_file = base + "_qiskit.py";

    // 1) Generating AST JSON
    std::string cmd_ast = "clang -Xclang -ast-dump=json -fsyntax-only " + c_file + " > " + json_file;
    std::cout << "Running: " << cmd_ast << "\n";
    if (std::system(cmd_ast.c_str()) != 0) {
        die("AST dump failed");
    }
    
    // 2) Generate raw MLIR
    std::string ir_exec = (backend_dir / "direct_ir_gen").string();
    if (!fs::exists(ir_exec)) {
        std::string cmd_compile = "clang++ -std=c++17 " + 
                                (backend_dir / "direct_ir_gen.cpp").string() + " " +
                                (exe_dir.parent_path() / "dialect/dialect.cpp").string() + " " +
                                (exe_dir.parent_path() / "dialect/utils.cpp").string() + 
                                " -o " + ir_exec;
        std::cout << "Compiling IR generator...\n";
        if (std::system(cmd_compile.c_str()) != 0) {
            die("Failed to compile IR generator.");
        }
    }

    std::string cmd_ir = ir_exec + " " + json_file + " " + mlir_file;
    std::cout << "Running: " << cmd_ir << "\n";
    if (std::system(cmd_ir.c_str()) != 0) {
        die("IR generation failed.");
    }

    // Create a copy of the original MLIR for comparison
    std::string original_mlir = base + ".original.mlir";
    std::string copy_cmd = "cp " + mlir_file + " " + original_mlir;
    std::system(copy_cmd.c_str());

    // 3) Optimize MLIR (quantum-safe passes)
    std::cout << "Optimizing MLIR with integrated passes...\n";
    if (!optimizeMlirFile(mlir_file, opt_mlir)) {
        die("MLIR optimization failed");
    }

    // Show a diff of the original vs optimized MLIR
    std::cout << "\nDiff between original and optimized MLIR:\n";
    std::string diff_cmd = "diff --color=always -u " + original_mlir + " " + opt_mlir;
    std::system(diff_cmd.c_str());
    
    // 4) Convert MLIR to Qiskit Python (original and optimized)
    std::string cmd_qiskit = "python3 " + (scripts_dir / "qmlir_to_qiskits.py").string() + " " + mlir_file + " " + qiskit_file;
    std::cout << "\nRunning: " << cmd_qiskit << "\n";
    if (std::system(cmd_qiskit.c_str()) != 0) {
        die("Qiskit translation failed");
    }

    // 5) Convert optimized MLIR to Qiskit Python 
    std::string opt_qiskit_file = base + ".opt_qiskit.py";
    std::string cmd_qiskit_opt = "python3 " + (scripts_dir / "qmlir_to_qiskits.py").string() + " " + opt_mlir + " " + opt_qiskit_file;
    std::cout << "Running: " << cmd_qiskit_opt << "\n";
    if (std::system(cmd_qiskit_opt.c_str()) != 0) {
        die("Qiskit translation failed.");
    }

    std::cout << "✅ Qiskit code generated: " << qiskit_file << "\n";
    std::cout << "✅ Optimized Qiskit code: " << opt_qiskit_file << "\n";
    
    // Count the number of quantum operations before and after optimization
    std::string count_ops_cmd = "echo 'Original gates:' && grep -c 'qc\\.' " + qiskit_file + 
                              " && echo 'Optimized gates:' && grep -c 'qc\\.' " + opt_qiskit_file;
    std::cout << "\n📊 Gate count comparison:\n";
    std::system(count_ops_cmd.c_str());

    return 0;
}

