// backend/passes/quantum_fusion_pass.cpp
#include "quantum_fusion_pass.h"
#include "EliminateAdjacentCx.h"
#include "EliminateAdjacentX.h"
#include "EliminateAdjacentCcx.h"
#include "DeadAllocRemoval.h"
#include "ConstantFolding.h"
#include "AncillaHoist.h"
#include "CommutativeCancellation.h"
#include "HighLevelFusion.h"

#include <fstream>
#include <iostream>
#include <sstream>

bool optimizeMlirFile(const std::string &inFile,
                      const std::string &outFile) {
  std::ifstream in(inFile);
  if (!in) return false;
  std::ostringstream buf; buf << in.rdbuf();
  std::string content = buf.str();

  int total = 0;
  total += eliminateAdjacentCx(content);
  total += eliminateAdjacentX(content);
  total += eliminateAdjacentCcx(content);
  total += removeDeadAllocs(content);
  total += foldInitSubZero(content);
  //total += hoistSingleAlloc(content);
  total += commuteCancelCx(content);
  total += fuseHighLevel(content);

  std::ofstream out(outFile);
  if (!out) return false;
  out << "// Quantum Fusion Optimization Summary:\n"
      << "// " << total << " total optimizations applied\n"
      << content;
  std::cout << "📊 Summary: " << total << " optimizations applied\n";
  return true;
}

