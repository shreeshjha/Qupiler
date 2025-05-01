#include <iostream>
#include <fstream>
#include <string> 
#include <vector>
#include <filesystem>
#include "../backend/passes/quantum_fusion_pass.h"

namespace fs = std::filesystem;

// Generate test case that will trigger specific optimizations

void generate_optimization_test_cases(const std::string& output_dir) {
    fs::create_directories(output_dir);

    // Test case 1: Adjacent CX gates 
    {
        std::ofstream out(output_dir + "/adjacent_cx_test.mlir");
        out << "func @adjacent_cx_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<2>\n";
        out << "  q.init %q0, 0 : i32\n";
        out << "  q.cx %q0[0], %q0[1]\n";
        out << "  q.cx %q0[0], %q0[1]\n";  // This should be eliminated
        out << "  %t0 = q.measure %q0 : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for EliminateAdjacentCx\n";
    }

    // Test case 2: Adjacent X gates 
    {
        std::ofstream out(output_dir + "/adjacent_x_test.mlir");
        out << "func @adjacent_x_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<2>\n";
        out << "  q.init %q0, 0 : i32\n";
        out << "  q.x %q0[0]\n";
        out << "  q.x %q0[0]\n";  // This should be eliminated
        out << "  %t0 = q.measure %q0 : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for EliminateAdjacentX\n";
    }

    // Test case 3: Adjacent CCX gates
    {
        std::ofstream out(output_dir + "/adjacent_ccx_test.mlir");
        out << "func @adjacent_ccx_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<3>\n";
        out << "  q.init %q0, 0 : i32\n";
        out << "  q.ccx %q0[0], %q0[1], %q0[2]\n";
        out << "  q.ccx %q0[0], %q0[1], %q0[2]\n";  // This should be eliminated
        out << "  %t0 = q.measure %q0 : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for EliminateAdjacentCcx\n";
    }
    
    // Test case 4: Dead alloc removal
    {
        std::ofstream out(output_dir + "/dead_alloc_test.mlir");
        out << "func @dead_alloc_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<2>\n";
        out << "  q.init %q0, 0 : i32\n";
        out << "  %q1 = q.alloc : !qreg<2>\n";  // This should be eliminated (unused)
        out << "  q.init %q0, 1 : i32\n";
        out << "  %t0 = q.measure %q0 : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for DeadAllocRemoval\n";
    }
    
    // Test case 5: Constant folding
    {
        std::ofstream out(output_dir + "/const_fold_test.mlir");
        out << "func @const_fold_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<2>\n";
        out << "  q.init %q0, 0 : i32\n";
        out << "  %q1 = q.alloc : !qreg<2>\n";
        out << "  q.init %q1, 0 : i32\n";
        out << "  q.sub %q1, %q0, %q0\n";  // Should be folded to q.init %q1, 0
        out << "  %t0 = q.measure %q1 : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for ConstantFolding\n";
    }
    
    // Test case 6: Extended constant folding
    {
        std::ofstream out(output_dir + "/ext_const_fold_test.mlir");
        out << "func @ext_const_fold_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<2>\n";
        out << "  q.init %q0, 0 : i32\n";
        out << "  %q1 = q.alloc : !qreg<2>\n";
        out << "  q.init %q1, 0 : i32\n";
        out << "  q.add %q1, %q0, %q0\n";  // Should be folded to q.init %q1, 0
        out << "  %t0 = q.measure %q1 : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for ExtendedConstantFolding\n";
    }
    
    // Test case 7: Identity removal
    {
        std::ofstream out(output_dir + "/identity_removal_test.mlir");
        out << "func @identity_removal_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<2>\n";
        out << "  q.init %q0, 0 : i32\n";
        out << "  q.cx %q0[0], %q0[0]\n";  // Should be removed (identity)
        out << "  q.ccx %q0[0], %q0[0], %q0[1]\n";  // Should be removed (identity)
        out << "  %t0 = q.measure %q0 : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for IdentityRemoval\n";
    }
    
    // Test case 8: Commutative cancellation
    {
        std::ofstream out(output_dir + "/commutative_test.mlir");
        out << "func @commutative_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<2>\n";
        out << "  q.init %q0, 0 : i32\n";
        out << "  q.cx %q0[0], %q0[1]\n";
        out << "  q.x %q0[0]\n";  // Independent operation
        out << "  q.cx %q0[0], %q0[1]\n";  // These should cancel out
        out << "  %t0 = q.measure %q0 : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for CommutativeCancellation\n";
    }

    // Test case 9: Ancilla hoisting
    {
        std::ofstream out(output_dir + "/ancilla_hoist_test.mlir");
        out << "func @ancilla_hoist_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<1>\n";
        out << "  q.init %q0, 0 : i32\n";
        out << "  %q1 = q.alloc : !qreg<1>\n";
        out << "  q.init %q1, 0 : i32\n";
        out << "  %q2 = q.alloc : !qreg<1>\n";
        out << "  q.init %q2, 0 : i32\n";
        out << "  q.cx %q0[0], %q1[0]\n";
        out << "  q.cx %q1[0], %q2[0]\n";
        out << "  %t0 = q.measure %q2 : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for AncillaHoist\n";
    }
}

// Run all tests and print results
void run_optimization_tests(const std::string& test_dir) {
    std::cout << "\nRunning optimization tests...\n";
    
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (entry.path().extension() == ".mlir") {
            std::string infile = entry.path().string();
            std::string outfile = entry.path().stem().string() + ".opt.mlir";
            
            std::cout << "\nTesting: " << entry.path().filename() << "\n";
            
            if (optimizeMlirFile(infile, outfile)) {
                // Read the optimized file to check what optimizations were applied
                std::ifstream in(outfile);
                std::string line;
                if (std::getline(in, line)) {
                    if (line.find("Quantum Fusion Optimization Summary") != std::string::npos) {
                        if (std::getline(in, line)) {
                            std::cout << line << "\n";
                        }
                    }
                }
            } else {
                std::cout << "Optimization failed!\n";
            }
        }
    }
}

int main(int argc, char** argv) {
    std::string test_dir = "opt_tests";
    
    std::cout << "Quantum Optimization Test Suite\n";
    std::cout << "===============================\n";
    
    // Generate test cases
    generate_optimization_test_cases(test_dir);
    
    // Run all tests
    run_optimization_tests(test_dir);
    
    return 0;
}

