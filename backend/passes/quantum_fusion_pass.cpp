// backend/passes/quantum_fusion_pass.cpp
#include "quantum_fusion_pass.h"
#include "EliminateAdjacentCx.h"
#include "EliminateAdjacentX.h"
#include "EliminateAdjacentCcx.h"
#include "IdentityRemoval.h"
#include "ExtendedConstantFolding.h"
#include "DeadAllocRemoval.h"
#include "ConstantFolding.h"
#include "AncillaHoist.h"
#include "CommutativeCancellation.h"
#include "HighLevelFusion.h"
#include "HighLevelFusionExtended.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <unordered_set>


bool optimizeMlirFile(const std::string &inFile,
                      const std::string &outFile) {
  std::ifstream in(inFile);
  if (!in) return false;
  std::ostringstream buf; buf << in.rdbuf();
  std::string content = buf.str();

  int total = 0;
  
  std::vector<std::string> applied;
  auto run_pass = [&](auto&& fn, const char* name) {
        int c = fn(content);
        if(c > 0) {
            total += c;
            applied.push_back(name);
        }
  };

  //total += eliminateAdjacentCx(content);
  //total += eliminateAdjacentX(content);
  //total += eliminateAdjacentCcx(content);
  //total += removeDeadAllocs(content);
  //total += foldInitSubZero(content);
  //total += foldAddZero(content);
  //total += removeIdentityOps(content);
  //total += fuseHighLevelExtended(content);
  //total += hoistSingleAlloc(content);
  //total += commuteCancelCx(content);
  //total += fuseHighLevel(content);

  run_pass(eliminateAdjacentCx,       "EliminateAdjacentCx");
  run_pass(eliminateAdjacentX,        "EliminateAdjacentX");
  run_pass(removeIdentityOps,         "IdentityRemoval");
  run_pass(eliminateAdjacentCcx,      "EliminateAdjacentCcx");
  run_pass(removeDeadAllocs,          "DeadAllocRemoval");
  run_pass(foldInitSubZero,           "ConstantFolding");
  run_pass(foldAddZero,               "ExtendedConstantFolding");
  run_pass(commuteCancelCx,           "CommutativeCancellation");
  run_pass(hoistSingleAlloc,          "AncillaHoist");
  run_pass(fuseExtendedRippleCarryAdder, "RippleCarryAdderFusion");
  //run_pass(fuseHighLevelExtended,     "HighLevelFusionExtended");
  run_pass(fuseHighLevel,             "HighLevelFusion");
  


  
  // Create a prepass optimization to identify potential patterns
  // This helps with creating testcases for optimizations
  auto analyze_patterns = [&]() {
    std::unordered_set<std::string> possiblePatterns;
    
    // Look for adjacent operations of the same type
    std::istringstream iss(content);
    std::string line, prevLine;
    std::string prevOp;
    
    while (std::getline(iss, line)) {
        if (line.find("q.cx") != std::string::npos) {
            if (prevOp == "q.cx") {
                possiblePatterns.insert("Adjacent CX gates");
            }
            prevOp = "q.cx";
        } 
        else if (line.find("q.x") != std::string::npos) {
            if (prevOp == "q.x") {
                possiblePatterns.insert("Adjacent X gates");
            }
            prevOp = "q.x";
        }
        else if (line.find("q.ccx") != std::string::npos) {
            if (prevOp == "q.ccx") {
                possiblePatterns.insert("Adjacent CCX gates");
            }
            prevOp = "q.ccx";
        }
        else {
            prevOp = "";
        }
        
        prevLine = line;
    }
    
    // Look for identity operations (CX with same control and target)
    iss.clear();
    iss.seekg(0);
    
    while (std::getline(iss, line)) {
        if (line.find("q.cx") != std::string::npos) {
            // Simplified check: look for patterns like %a[i], %a[i]
            size_t pos = line.find("q.cx");
            if (pos != std::string::npos) {
                std::string rest = line.substr(pos + 4);
                size_t firstVar = rest.find("%");
                if (firstVar != std::string::npos) {
                    size_t comma = rest.find(",", firstVar);
                    if (comma != std::string::npos) {
                        std::string firstOperand = rest.substr(firstVar, comma - firstVar);
                        std::string secondHalf = rest.substr(comma + 1);
                        if (secondHalf.find(firstOperand) != std::string::npos) {
                            possiblePatterns.insert("Identity CX operations");
                        }
                    }
                }
            }
        }
    }
    
    // Print potential optimization opportunities
    if (!possiblePatterns.empty()) {
        std::cout << "Potential optimization opportunities detected:\n";
        for (const auto& pattern : possiblePatterns) {
            std::cout << "  - " << pattern << "\n";
        }
    }
  };
  
  analyze_patterns();
  
  std::ofstream out(outFile);
  if (!out) return false;
  // build a comma-separated list
  std::ostringstream list;
  for (size_t i = 0; i < applied.size(); ++i) {
    if (i) list << ", ";
    list << applied[i];
  }
  out << "// Quantum Fusion Optimization Summary:\n"
      << "// " << total << " optimizations applied: " << list.str() << "\n"
      << content;
    std::cout << "📊 Summary: " << total
            << " optimizations applied [" << list.str() << "]\n";
  return true;
}

