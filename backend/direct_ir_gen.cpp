#include "qmlir_ir.hpp"
#include "json.hpp"
#include "../dialect/dialect.hpp"
#include "../dialect/utils.hpp"

#include <unordered_map>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <functional>

constexpr int QBIT_WIDTH = 4;

class IRGenerator {
private:
    QMLIR_Function& func;
    std::unordered_map<std::string, std::string>& vars;
    bool debug;
    bool quantum_mode = false;

public:
    IRGenerator(QMLIR_Function& f, std::unordered_map<std::string, std::string>& v, bool d = false)
        : func(f), vars(v), debug(d) {}

    void process_function(const nlohmann::json& func_json) {
        if (func_json.contains("name") && func_json["name"].is_string()) {
            func.name = func_json["name"].get<std::string>();
            if (debug) std::cout << "Function name: " << func.name << std::endl;
            if (func.name == "quantum_circuit") quantum_mode = true;
        }
        if (func_json.contains("inner") && func_json["inner"].is_array()) {
            for (const auto& inner : func_json["inner"]) {
                process_node(inner);
            }
        }
    }

    void process_compound_stmt(const nlohmann::json& compound_json) {
        if (compound_json.contains("inner") && compound_json["inner"].is_array()) {
            for (const auto& stmt : compound_json["inner"]) {
                process_node(stmt);
            }
        }
    }
    void process_decl_stmt(const nlohmann::json& decl_json) {
        if (!decl_json.contains("inner") || !decl_json["inner"].is_array()) return;
        for (const auto& decl : decl_json["inner"]) {
            if (!decl.is_object() || decl["kind"] != "VarDecl") continue;
            std::string var_name = decl.value("name", "");
            if (var_name.empty()) continue;
            std::string tmp = new_tmp(quantum_mode ? "q" : "t");
            vars[var_name] = tmp;
            if (quantum_mode) {
                emit_qubit_alloc(func, tmp, QBIT_WIDTH);
                if (decl.contains("inner") && !decl["inner"].empty()) {
                    const auto& init = decl["inner"][0];
                    std::function<bool(const nlohmann::json&, std::string&)> find_ref =
                            [&](const nlohmann::json& node, std::string& var) -> bool {
                                if (node["kind"] == "DeclRefExpr" && node.contains("referencedDecl")) {
                                    var = node["referencedDecl"].value("name", "");
                                    return !var.empty();
                                }
                                if (node.contains("inner")) {
                                    for (const auto& inner : node["inner"]) {
                                        if (find_ref(inner, var))
                                            return true;
                                    }
                                }
                                return false;
                            };
                    if (init["kind"] == "IntegerLiteral" && init.contains("value")) {
                        int value = std::stoi(init["value"].get<std::string>());
                        emit_qubit_init(func, tmp, value, QBIT_WIDTH);
                    } else if (init["kind"] == "BinaryOperator") {
                        std::string op = init.value("opcode", "");
                        std::string left_var, right_var;
                        /*std::function<bool(const nlohmann::json&, std::string&)> find_ref =
                            [&](const nlohmann::json& node, std::string& var) -> bool {
                                if (node["kind"] == "DeclRefExpr" && node.contains("referencedDecl")) {
                                    var = node["referencedDecl"].value("name", "");
                                    return !var.empty();
                                }
                                if (node.contains("inner")) {
                                    for (const auto& inner : node["inner"]) {
                                        if (find_ref(inner, var))
                                            return true;
                                    }
                                }
                                return false;
                            };*/
                        if (init.contains("inner") && init["inner"].size() >= 2 &&
                            find_ref(init["inner"][0], left_var) &&
                            find_ref(init["inner"][1], right_var) &&
                            vars.count(left_var) && vars.count(right_var)) {
                            std::string result = new_tmp("q");
                            emit_qubit_alloc(func, result, QBIT_WIDTH);
                            if (op == "+")
                                emit_quantum_adder(func, result, vars[left_var], vars[right_var], QBIT_WIDTH);
                            else if (op == "-")
                                emit_quantum_subtractor(func, result, vars[left_var], vars[right_var], QBIT_WIDTH);
                            else if (op == "*")
                                emit_quantum_multiplier(func, result, vars[left_var], vars[right_var], QBIT_WIDTH);
                            else if (op == "/")
                                emit_quantum_divider(func, result, vars[left_var], vars[right_var], QBIT_WIDTH);
                            else if (op == "%")
                                emit_quantum_modulo(func, result, vars[left_var], vars[right_var], QBIT_WIDTH);
                            else if (op == "&&")
                                emit_quantum_and(func, result, vars[left_var], vars[right_var], QBIT_WIDTH);
                            vars[var_name] = result;
                        }
                    } else if (init["kind"] == "UnaryOperator") {
                        std::string op = init.value("opcode", "");
                        std::string var;
                        if (find_ref(init["inner"][0], var) && vars.count(var)) {
                            if (op == "-") {
                                if (quantum_mode)
                                    emit_quantum_negate(func, tmp, vars[var], QBIT_WIDTH);
                                else
                                    func.ops.push_back({QOpKind::Neg, tmp, vars[var]});
                            }    
                            else if (op == "++") {
                                if (quantum_mode)
                                    emit_quantum_increment(func, tmp, vars[var], QBIT_WIDTH);
                                else
                                    func.ops.push_back({QOpKind::Inc, tmp, vars[var]});
                                } 
                            else if (op == "--") {
                                if (quantum_mode)
                                    emit_quantum_decrement(func, tmp, vars[var], QBIT_WIDTH);
                                else
                                    func.ops.push_back({QOpKind::Dec, tmp, vars[var]});
                            }
                        }
                    }
                }
            } else {
                if (decl.contains("inner") && !decl["inner"].empty()) {
                    const auto& init = decl["inner"][0];
                    if (init["kind"] == "IntegerLiteral" && init.contains("value")) {
                        int value = std::stoi(init["value"].get<std::string>());
                        func.ops.push_back({QOpKind::Const, tmp, "", "", value});
                    } else if (init["kind"] == "BinaryOperator") {
                        std::string op = init.value("opcode", "");
                        std::string left_var, right_var;
                        std::function<bool(const nlohmann::json&, std::string&)> find_ref =
                            [&](const nlohmann::json& node, std::string& var) -> bool {
                                if (node["kind"] == "DeclRefExpr" && node.contains("referencedDecl")) {
                                    var = node["referencedDecl"].value("name", "");
                                    return !var.empty();
                                }
                                if (node.contains("inner")) {
                                    for (const auto& inner : node["inner"]) {
                                        if (find_ref(inner, var))
                                            return true;
                                    }
                                }
                                return false;
                            };
                        if (init.contains("inner") && init["inner"].size() >= 2 &&
                            find_ref(init["inner"][0], left_var) &&
                            find_ref(init["inner"][1], right_var) &&
                            vars.count(left_var) && vars.count(right_var)) {
                            if (op == "+")
                                func.ops.push_back({QOpKind::Add, tmp, vars[left_var], vars[right_var]});
                            else if (op == "-")
                                func.ops.push_back({QOpKind::Sub, tmp, vars[left_var], vars[right_var]});
                            else if (op == "*")
                                func.ops.push_back({QOpKind::Mul, tmp, vars[left_var], vars[right_var]});
                            else if (op == "/")
                                func.ops.push_back({QOpKind::Div, tmp, vars[left_var], vars[right_var]});
                            else if (op == "%")
                                func.ops.push_back({QOpKind::Mod, tmp, vars[left_var], vars[right_var]});
                        }
                    }
                }
            }
        }
    }

    void process_call_expr(const nlohmann::json& call_json) {
        if (!call_json.contains("inner")) return;
        for (size_t i = 1; i < call_json["inner"].size(); ++i) {
            std::string var_name;
            std::function<bool(const nlohmann::json&, std::string&)> find_ref =
                [&](const nlohmann::json& node, std::string& var) -> bool {
                    if (node["kind"] == "DeclRefExpr" && node.contains("referencedDecl")) {
                        var = node["referencedDecl"].value("name", "");
                        return !var.empty();
                    }
                    if (node.contains("inner")) {
                        for (const auto& inner : node["inner"]) {
                            if (find_ref(inner, var))
                                return true;
                        }
                    }
                    return false;
                };
            if (find_ref(call_json["inner"][i], var_name) && vars.count(var_name)) {
                if (quantum_mode) {
                    std::string measured = new_tmp("t");
                    emit_measure(func, vars[var_name], measured);
                    func.ops.push_back({QOpKind::Print, "", measured, ""});
                } else {
                    func.ops.push_back({QOpKind::Print, "", vars[var_name], ""});
                }
                break;
            }
        }
    }

    void process_return_stmt(const nlohmann::json&) {
        func.ops.push_back({QOpKind::Return});
    }

    void process_node(const nlohmann::json& node) {
        std::string kind = node.value("kind", "");
        if (kind == "FunctionDecl") process_function(node);
        else if (kind == "CompoundStmt") process_compound_stmt(node);
        else if (kind == "DeclStmt") process_decl_stmt(node);
        else if (kind == "CallExpr") process_call_expr(node);
        else if (kind == "ReturnStmt") process_return_stmt(node);
    }

    void generate_ir_from_json(const nlohmann::json& json) {
        if (json.contains("kind") && json["kind"] == "FunctionDecl") {
            process_function(json);
        } else if (json.is_array() || json.is_object()) {
            if (json.contains("inner") && json["inner"].is_array()) {
                for (const auto& node : json["inner"]) {
                    process_node(node);
                }
            } else {
                process_node(json);
            }
        }
        if (func.ops.empty() || func.ops.back().kind != QOpKind::Return) {
            func.ops.push_back({QOpKind::Return});
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_json> <output_mlir> [debug]\\n";
        return 1;
    }
    bool debug = (argc > 3 && std::string(argv[3]) == "debug");
    try {
        std::ifstream in(argv[1]);
        if (!in) throw std::runtime_error("Failed to open input file.");
        nlohmann::json json_data;
        in >> json_data;
        QMLIR_Function fn;
        std::unordered_map<std::string, std::string> vars;
        IRGenerator generator(fn, vars, debug);
        generator.generate_ir_from_json(json_data);
        std::ofstream out(argv[2]);
        if (!out) throw std::runtime_error("Failed to open output file.");
        fn.emit(out);
        std::cout << "MLIR generated successfully.\\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\\n";
        return 1;
    }
}

