#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./main <source_file.c>\n";
        return 1;
    }

    std::string c_file = argv[1];
    if (!fs::exists(c_file)) {
        std::cerr << "Error: File " << c_file << " does not exist.\n";
        return 1;
    }

    std::string base = fs::path(c_file).stem();
    std::string json_file = base + ".json";
    std::string mlir_file = base + ".mlir";
    std::string qiskit_file = base + "_qiskit.py";

    std::string cmd_ast = "clang -Xclang -ast-dump=json -fsyntax-only " + c_file + " > " + json_file;
    std::cout << "Running: " << cmd_ast << "\n";
    if (std::system(cmd_ast.c_str()) != 0) {
        std::cerr << "Failed to generate JSON AST.\n";
        return 1;
    }

    std::string ir_exec = "../backend/direct_ir_gen";
    if (!fs::exists(ir_exec)) {
        std::string cmd_compile = "clang++ -std=c++17 ../backend/direct_ir_gen.cpp ../dialect/dialect.cpp ../dialect/utils.cpp -o " + ir_exec;
        std::cout << "Compiling IR generator...\n";
        if (std::system(cmd_compile.c_str()) != 0) {
            std::cerr << "Failed to compile IR generator.\n";
            return 1;
        }
    }

    std::string cmd_ir = ir_exec + " " + json_file + " " + mlir_file;
    std::cout << "Running: " << cmd_ir << "\n";
    if (std::system(cmd_ir.c_str()) != 0) {
        std::cerr << "IR generation failed.\n";
        return 1;
    }

    std::string cmd_qiskit = "python3 ../scripts/qmlir_to_qiskits.py " + mlir_file + " " + qiskit_file;
    std::cout << "Running: " << cmd_qiskit << "\n";
    if (std::system(cmd_qiskit.c_str()) != 0) {
        std::cerr << "Qiskit translation failed.\n";
        return 1;
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
        std::cerr << "Failed to open generated Qiskit file.\n";
    }

    return 0;
}

