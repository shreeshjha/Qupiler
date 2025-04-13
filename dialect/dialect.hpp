#pragma once
#include "../backend/qmlir_ir.hpp"
#include <string>

void emit_qubit_alloc(QMLIR_Function& func, const std::string& tmp, int size);
void emit_qubit_init(QMLIR_Function& func, const std::string& qubit_tmp, int value, int size);
void emit_measure(QMLIR_Function& func, const std::string& qubit_tmp, const std::string& result_tmp);

void emit_quantum_adder(QMLIR_Function& func, const std::string& result,
                        const std::string& a, const std::string& b, int num_bits);

void emit_quantum_subtractor(QMLIR_Function& func, const std::string& result,
                             const std::string& a, const std::string& b, int num_bits);



void emit_quantum_shift(QMLIR_Function& func, const std::string& src,const std::string& dst, int shift, int num_bits);

void emit_quantum_multiplier(QMLIR_Function& func, const std::string& result, 
                           const std::string& a, const std::string& b, int num_bits);

void emit_quantum_divider(QMLIR_Function& func,
                          const std::string& result,
                          const std::string& a,
                          const std::string& b,
                          int num_bits);

void emit_quantum_modulo(QMLIR_Function& func, const std::string& result, const std::string& a, const std::string& b, int num_bits);

void emit_quantum_negate(QMLIR_Function& func, const std::string& result, 
                          const std::string& input, int num_bits);

void emit_quantum_increment(QMLIR_Function& func, const std::string& result, 
                            const std::string& input, int num_bits);

void emit_quantum_decrement(QMLIR_Function& func, const std::string& result, 
                            const std::string& input, int num_bits);

void emit_quantum_and(QMLIR_Function& func, const std::string& result,
                      const std::string& a, const std::string& b, int num_bits);


void emit_quantum_or(QMLIR_Function& func, const std::string& result,
                     const std::string& a, const std::string& b, int num_bits);

void reset_global_tmps();
