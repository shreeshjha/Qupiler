#ifndef QUANTUM_FUSION_PASS_H
#define QUANTUM_FUSION_PASS_H

#include <string>

// Simple function to optimize an MLIR file by removing adjacent identical CNOT gates
bool optimizeMlirFile(const std::string &inputFile, const std::string &outputFile);

#endif // QUANTUM_FUSION_PASS_H
