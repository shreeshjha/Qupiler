// backend/passes/quantum_fusion_pass.h
#ifndef QUANTUM_FUSION_PASS_H
#define QUANTUM_FUSION_PASS_H

#include <string>

bool optimizeMlirFile(const std::string &inputFile,
                      const std::string &outputFile);

#endif // QUANTUM_FUSION_PASS_H

