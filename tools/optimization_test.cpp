#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <unordered_map>
#include <algorithm>
#include <regex>
#include "../backend/passes/quantum_fusion_pass.h"

// Declare the individual optimization functions if not directly accessible
// We're just declaring them here - they're implemented in your optimization pass files
extern int eliminateAdjacentCx(std::string &content);
extern int eliminateAdjacentX(std::string &content);
extern int eliminateAdjacentCcx(std::string &content);
extern int removeIdentityOps(std::string &content);
extern int foldInitSubZero(std::string &content);
extern int foldAddZero(std::string &content);
extern int removeDeadAllocs(std::string &content);
extern int commuteCancelCx(std::string &content);
extern int hoistSingleAlloc(std::string &content);
extern int fuseHighLevel(std::string &content);
extern int fuseHighLevelExtended(std::string &content);

namespace fs = std::filesystem;

// Detailed logging for debugging optimization passes
bool verbose_logging = true;

// Structure to track optimization results
struct OptimizationResult {
    std::string test_name;
    std::string optimizations_applied;
    int count;
    bool success;
};


// Function prototypes
void generate_optimization_test_cases(const std::string& output_dir);
std::vector<OptimizationResult> run_optimization_tests(const std::string& test_dir);
void print_optimization_summary(const std::vector<OptimizationResult>& results);

// Generate test cases that will trigger specific optimizations
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
        out << "  q.init %q0[0], 0 : i32\n";
        out << "  %q1 = q.alloc : !qreg<2>\n";
        out << "  q.sub %q1, %q0[0], %q0[0]\n";  // Should be folded to q.init %q1, 0
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
        out << "  q.init %q0[0], 0 : i32\n";
        out << "  %q1 = q.alloc : !qreg<2>\n";
        out << "  q.add %q1, %q0[0], %q0[0]\n";  // Should be folded to q.init %q1, 0
        out << "  %t0 = q.measure %q1 : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for ExtendedConstantFolding\n";
    }
    
    // Test case 7: Identity removal (improved)
    {
        std::ofstream out(output_dir + "/identity_removal_test.mlir");
        out << "func @identity_removal_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<2>\n";
        out << "  q.init %q0, 0 : i32\n";
        out << "  q.cx %q0[0], %q0[0]\n";  // Should be removed (identity CX)
        out << "  q.ccx %q0[0], %q0[0], %q0[1]\n";  // Should be removed (duplicate controls)
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

    // Test case 9: Ancilla hoisting (improved)
    {
        std::ofstream out(output_dir + "/ancilla_hoist_test.mlir");
        out << "func @ancilla_hoist_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<1>\n";
        out << "  %q1 = q.alloc : !qreg<1>\n";
        out << "  %q2 = q.alloc : !qreg<1>\n";
        out << "  %q3 = q.alloc : !qreg<1>\n";
        out << "  q.init %q0, 0 : i32\n";
        out << "  q.init %q1, 0 : i32\n";
        out << "  q.init %q2, 0 : i32\n";
        out << "  q.init %q3, 0 : i32\n";
        out << "  q.cx %q0[0], %q1[0]\n";
        out << "  q.cx %q1[0], %q2[0]\n";
        out << "  q.cx %q2[0], %q3[0]\n";
        out << "  %t0 = q.measure %q3 : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for AncillaHoist\n";
    }
    
    // NEW Test case 10: Direct tests for ExtendedConstantFolding
    {
        std::ofstream out(output_dir + "/add_zero_test.mlir");
        out << "func @add_zero_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<2>\n";
        out << "  q.init %q0[0], 0 : i32\n";  // Initialize to 0
        out << "  %q1 = q.alloc : !qreg<2>\n";
        out << "  q.add %q1, %q0[0], %q0[0]\n";  // q1 = 0 + 0, should fold to init q1, 0
        out << "  %t0 = q.measure %q1 : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for AddZero (ExtendedConstantFolding)\n";
    }
    
    // NEW Test case 11: High-Level Fusion
    {
        std::ofstream out(output_dir + "/high_level_fusion_test.mlir");
        out << "func @high_level_fusion_test() -> () {\n";
        out << "  %q0 = q.alloc : !qreg<2>\n";
        out << "  %q1 = q.alloc : !qreg<2>\n";
        out << "  %r = q.alloc : !qreg<2>\n";
        out << "  q.init %q0, 1 : i32\n";
        out << "  q.init %q1, 2 : i32\n";
        out << "  q.cx %q0[0], %r[0]\n";   // 1-bit adder pattern
        out << "  q.cx %q1[0], %r[0]\n";
        out << "  q.ccx %q0[0], %q1[0], %r[1]\n";
        out << "  %t0 = q.measure %r : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for HighLevelFusion\n";
    }

    // Test case 12: Extended ripple-carry adder fusion
    {
        std::ofstream out(output_dir + "/high_level_fusion_ext_test.mlir");
        out << "func @high_level_fusion_ext_test() -> () {\n";
        out << "  %a = q.alloc : !qreg<2>\n";
        out << "  %b = q.alloc : !qreg<2>\n";
        out << "  %r = q.alloc : !qreg<3>\n";
        out << "  q.init %a[0], 1 : i32\n";
        out << "  q.init %a[1], 0 : i32\n";
        out << "  q.init %b[0], 1 : i32\n";
        out << "  q.init %b[1], 1 : i32\n";
        out << "  q.init %r[0], 0 : i32\n";
        out << "  q.init %r[1], 0 : i32\n";
        out << "  q.init %r[2], 0 : i32\n";
        out << "  q.cx %a[0], %r[0]\n";
        out << "  q.cx %b[0], %r[0]\n";
        out << "  q.ccx %a[0], %b[0], %r[1]\n";
        out << "  q.cx %a[1], %r[1]\n";
        out << "  q.cx %b[1], %r[1]\n";
        out << "  q.ccx %a[1], %b[1], %r[2]\n";
        out << "  %t0 = q.measure %r : !qreg -> i32\n";
        out << "  q.print %t0\n";
        out << "  return\n";
        out << "}\n";
        std::cout << "Created test for HighLevelFusionExtended\n";
    }
}

// Run all tests and collect results
/*
std::vector<OptimizationResult> run_optimization_tests(const std::string& test_dir) {
    std::cout << "\nRunning optimization tests...\n";
    
    std::vector<OptimizationResult> results;
    
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (entry.path().extension() == ".mlir") {
            std::string infile = entry.path().string();
            std::string outfile = entry.path().stem().string() + ".opt.mlir";
            
            std::cout << "\nTesting: \"" << entry.path().filename() << "\"\n";
            
            OptimizationResult result;
            result.test_name = entry.path().filename().string();
            result.count = 0;
            result.success = false;
            result.optimizations_applied = "";
            
            if (optimizeMlirFile(infile, outfile)) {
                // Read the optimized file to check what optimizations were applied
                std::ifstream in(outfile);
                std::string line;
                std::string summary_line;
                
                // Read the first line which should be the summary header
                if (std::getline(in, line) && line.find("Quantum Fusion Optimization Summary") != std::string::npos) {
                    // Second line contains the actual summary
                    if (std::getline(in, summary_line)) {
                        std::cout << summary_line << "\n";
                        
                        // First, extract the raw count of optimizations
                        size_t countStart = summary_line.find(": ") + 2;
                        if (countStart != std::string::npos + 2) {
                            size_t countEnd = summary_line.find(" optimizations");
                            if (countEnd != std::string::npos) {
                                std::string countStr = summary_line.substr(countStart, countEnd - countStart);
                                try {
                                    result.count = std::stoi(countStr);
                                } catch(...) {
                                    std::cerr << "Error parsing optimization count: " << countStr << "\n";
                                }
                            }
                        }
                        
                        // Second, extract the list of applied optimizations
                        size_t namesStart = summary_line.find("applied: ");
                        if (namesStart != std::string::npos) {
                            namesStart += 9; // Length of "applied: "
                            result.optimizations_applied = summary_line.substr(namesStart);
                            // Remove trailing whitespace or punctuation
                            result.optimizations_applied.erase(result.optimizations_applied.find_last_not_of(" \n\r\t.") + 1);
                        }
                        
                        // Consider the test successful if at least one optimization was applied
                        result.success = (result.count > 0);
                    }
                }
                
                results.push_back(result);
            } else {
                std::cout << "Optimization failed!\n";
                result.optimizations_applied = "ERROR";
                results.push_back(result);
            }
        }
    }
    
    return results;
}
*/ 

// Run all tests and collect results
std::vector<OptimizationResult> run_optimization_tests(const std::string& test_dir) {
    std::cout << "\nRunning optimization tests...\n";
    
    std::vector<OptimizationResult> results;
    
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (entry.path().extension() == ".mlir") {
            std::string infile = entry.path().string();
            std::string outfile = entry.path().stem().string() + ".opt.mlir";
            
            std::cout << "\nTesting: \"" << entry.path().filename() << "\"\n";
            
            OptimizationResult result;
            result.test_name = entry.path().filename().string();
            result.count = 0;
            result.success = false;
            result.optimizations_applied = "";
            
            if (optimizeMlirFile(infile, outfile)) {
                // Read the optimized file to check what optimizations were applied
                std::ifstream in(outfile);
                std::string line;
                std::string summary_line;
                
                // Read the first line which should be the summary header
                if (std::getline(in, line) && line.find("Quantum Fusion Optimization Summary") != std::string::npos) {
                    // Second line contains the actual summary
                    if (std::getline(in, summary_line)) {
                        std::cout << summary_line << "\n";
                        
                        // Parse the optimization count without using regex
                        size_t count_start = summary_line.find("// ") + 3;
                        if (count_start != std::string::npos + 3) {
                            size_t count_end = summary_line.find(" optimizations");
                            if (count_end != std::string::npos) {
                                std::string count_str = summary_line.substr(count_start, count_end - count_start);
                                try {
                                    result.count = std::stoi(count_str);
                                } catch(...) {
                                    std::cerr << "Error parsing optimization count\n";
                                }
                            }
                        }
                        
                        // Extract the list of applied optimizations without using regex
                        size_t applied_start = summary_line.find("applied: ");
                        if (applied_start != std::string::npos) {
                            applied_start += 9; // Length of "applied: "
                            result.optimizations_applied = summary_line.substr(applied_start);
                        }
                        
                        // Consider the test successful if at least one optimization was applied
                        result.success = (result.count > 0);
                    }
                }
                
                results.push_back(result);
            } else {
                std::cout << "Optimization failed!\n";
                result.optimizations_applied = "ERROR";
                results.push_back(result);
            }
        }
    }
    
    return results;
}

// Print a detailed summary of optimization results
void print_optimization_summary(const std::vector<OptimizationResult>& results) {
    std::cout << "\n===================================\n";
    std::cout << "Optimization Test Summary\n";
    std::cout << "===================================\n";
    
    int total_passed = 0;
    std::unordered_map<std::string, int> optimization_counts;
    
    for (const auto& result : results) {
        std::string status = result.success ? "✅ PASS" : "❌ FAIL";
        std::cout << status << " | " << result.test_name << " | ";
        
        if (result.optimizations_applied == "ERROR") {
            std::cout << "Error running optimization\n";
        } else if (result.count == 0 || result.optimizations_applied.empty()) {
            std::cout << "No optimizations applied\n";
        } else {
            std::cout << result.count << " optimizations: " << result.optimizations_applied << "\n";
            
            // Extract individual optimization names for counting
            std::string ops = result.optimizations_applied;
            size_t pos = 0;
            while ((pos = ops.find(", ")) != std::string::npos) {
                optimization_counts[ops.substr(0, pos)]++;
                ops.erase(0, pos + 2);
            }
            if (!ops.empty()) {
                optimization_counts[ops]++;
            }
            
            total_passed++;
        }
    }
    
    std::cout << "\nPassed " << total_passed << " out of " << results.size() << " tests\n\n";
    
    std::cout << "Optimization Effectiveness:\n";
    std::cout << "--------------------------\n";
    for (const auto& [opt, count] : optimization_counts) {
        std::cout << "  " << opt << ": " << count << " applications\n";
    }
    
    // Check for missing optimizations
    std::vector<std::string> all_optimizations = {
        "EliminateAdjacentCx",
        "EliminateAdjacentX",
        "EliminateAdjacentCcx",
        "IdentityRemoval",
        "ConstantFolding",
        "ExtendedConstantFolding",
        "DeadAllocRemoval",
        "CommutativeCancellation",
        "AncillaHoist",
        "HighLevelFusion",
        "HighLevelFusionExtended"
    };
    
    std::cout << "\nUnused Optimizations:\n";
    std::cout << "--------------------\n";
    for (const auto& opt : all_optimizations) {
        if (optimization_counts.find(opt) == optimization_counts.end()) {
            std::cout << "  " << opt << " is not being used in any test\n";
        }
    }
}

int main(int argc, char** argv) {
    std::string test_dir = "opt_tests";
    
    std::cout << "Quantum Optimization Test Suite\n";
    std::cout << "===============================\n";
    
    // Process command-line options
    bool debug_mode = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug" || arg == "-d") {
            debug_mode = true;
        } else if (arg == "--verbose" || arg == "-v") {
            verbose_logging = true;
        } else if (arg == "--dir" && i + 1 < argc) {
            test_dir = argv[++i];
        }
    }
    
    // Generate test cases
    generate_optimization_test_cases(test_dir);
    
    // Run all tests
    std::vector<OptimizationResult> results = run_optimization_tests(test_dir);
    
    // Print test summary
    print_optimization_summary(results);
    
    // If in debug mode, run extra diagnostics
    if (debug_mode) {
        std::cout << "\nDebugging optimization passes...\n";
        std::cout << "\nWARNING: Direct function calls disabled due to linker issues.\n";
        std::cout << "Add individual pass tests by modifying quantum_fusion_pass.cpp instead.\n";
        
        // Suggest adding debugging to quantum_fusion_pass.cpp
        std::cout << "\nTo debug individual passes, modify quantum_fusion_pass.cpp to add this code:\n\n";
        std::cout << "auto run_pass = [&](auto&& fn, const char* name) {\n";
        std::cout << "    std::cout << \"Running pass: \" << name << \"...\\n\";\n";
        std::cout << "    int c = fn(content);\n";
        std::cout << "    std::cout << \"Result: \" << c << \" optimizations\\n\";\n";
        std::cout << "    if(c > 0) {\n";
        std::cout << "        total += c;\n";
        std::cout << "        applied.push_back(name);\n";
        std::cout << "    }\n";
        std::cout << "};\n";
    }
    
    return 0;
}
