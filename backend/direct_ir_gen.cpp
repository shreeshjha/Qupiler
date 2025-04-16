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

    
    void process_while_stmt(const nlohmann::json& while_json) {
    // Extract condition and body
    const auto& condition = while_json["inner"][0];
    const auto& body = while_json["inner"][1];
    
    if (quantum_mode) {
        // QUANTUM MODE - We need a loop unrolling approach for fixed iterations
        
        // Step 1: Create a quantum register to store the condition result
        std::string cond_reg = new_tmp("cond");
        emit_qubit_alloc(func, cond_reg, 1);  // 1 qubit for boolean result
        
        // Step 2: Fix maximum iterations to avoid infinite loops
        int max_iterations = QBIT_WIDTH;  // Default max iterations equals register width
        
        // Step 3: Create a new register to track the current value of the loop variable
        std::string current_var;
        std::string updated_var;
        std::string target_var;
        
        // Find the variable that gets modified in the loop
        std::function<void(const nlohmann::json&)> find_modified_var =
            [&](const nlohmann::json& node) {
                if (node["kind"] == "BinaryOperator" && node.value("opcode", "") == "=") {
                    std::string var_name;
                    if (node.contains("inner") && node["inner"].size() >= 1 &&
                        node["inner"][0]["kind"] == "DeclRefExpr" &&
                        node["inner"][0].contains("referencedDecl")) {
                        var_name = node["inner"][0]["referencedDecl"].value("name", "");
                        if (!var_name.empty() && vars.count(var_name)) {
                            target_var = var_name;
                        }
                    }
                }
                if (node.contains("inner")) {
                    for (const auto& inner : node["inner"]) {
                        find_modified_var(inner);
                    }
                }
            };
        
        find_modified_var(body);
        
        if (!target_var.empty()) {
            // Start with the initial value of the variable
            current_var = vars[target_var];
            
            // Unroll the while loop for max_iterations
            for (int i = 0; i < max_iterations; i++) {
                if (debug) std::cout << "Unrolling iteration " << i << std::endl;
                
                // Step 1: Evaluate condition x < y and store in cond_reg[0]
                if (condition["kind"] == "BinaryOperator") {
                    std::string op = condition.value("opcode", "");
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

                    if (condition.contains("inner") && condition["inner"].size() >= 2 &&
                        find_ref(condition["inner"][0], left_var) &&
                        find_ref(condition["inner"][1], right_var) &&
                        vars.count(left_var) && vars.count(right_var)) {
                        
                        // For this iteration, update the variable references if needed
                        std::string left_reg = left_var == target_var ? current_var : vars[left_var];
                        std::string right_reg = right_var == target_var ? current_var : vars[right_var];
                        
                        // Different comparisons require different quantum circuits
                        if (op == "<") {
                            std::string diff = new_tmp("diff"); 
                            emit_qubit_alloc(func, diff, QBIT_WIDTH); 
                            emit_quantum_subtractor(func, diff, right_reg, left_reg, QBIT_WIDTH);
                            std::string nz = new_tmp("nz");
                            emit_qubit_alloc(func, nz, 1);

                            for(int j = 0; j < QBIT_WIDTH; j++) {
                                func.ops.push_back({
                                    QOpKind::Custom, "",
                                    diff + "[" + std::to_string(j) + "]",
                                    nz + "[0]",
                                    0, "q.cx"
                                });
                            }

                            // 2. Compute the negated sign of diff.
                            std::string not_msb = new_tmp("not_msb");
                            emit_qubit_alloc(func, not_msb, 1);
                            // Copy the MSB:
                            func.ops.push_back({
                                QOpKind::Custom, "",
                                diff + "[" + std::to_string(QBIT_WIDTH - 1) + "]",
                                not_msb + "[0]",
                                0, "q.cx"
                            });
                            // Invert the copied bit (so that not_msb is 1 when diff is nonnegative):
                            func.ops.push_back({
                                QOpKind::Custom, "",
                                not_msb + "[0]",
                                "",
                                0, "q.x"
                            });

                            // 3. Combine the nonzero flag with the not_msb using an AND (Toffoli gate).
                            // Prepare cond_reg (which is the loop condition register) as our target.
                            // Ensure it starts in state 0, then flip it if both nz and not_msb are 1.
                            func.ops.push_back({
                                QOpKind::Custom, cond_reg + "[0]",
                                nz + "[0]",
                                not_msb + "[0]",
                                0, "q.ccx"
                            });
                        }
                    }
                }
                
                // Step 2: Process body conditionally using cond_reg[0] as control
                // We need to find and process the assignment statement that updates the variable
                if (body["kind"] == "CompoundStmt" && body.contains("inner") && body["inner"].is_array()) {
                    for (const auto& stmt : body["inner"]) {
                        if (stmt["kind"] == "BinaryOperator" && stmt.value("opcode", "") == "=") {
                            std::string assigned_var;
                            if (stmt["inner"][0]["kind"] == "DeclRefExpr" && 
                                stmt["inner"][0].contains("referencedDecl")) {
                                assigned_var = stmt["inner"][0]["referencedDecl"].value("name", "");
                            }
                            
                            if (assigned_var == target_var) {
                                // Handle the right-hand side expression
                                // For this example we handle operations like x = x + 1
                                if (stmt["inner"][1]["kind"] == "BinaryOperator") {
                                    std::string op = stmt["inner"][1].value("opcode", "");
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
                                    
                                    if (stmt["inner"][1].contains("inner") && 
                                        stmt["inner"][1]["inner"].size() >= 2 &&
                                        find_ref(stmt["inner"][1]["inner"][0], left_var) &&
                                        find_ref(stmt["inner"][1]["inner"][1], right_var)) {
                                        
                                        // Update variable references if needed
                                        std::string left_reg = left_var == target_var ? current_var : vars[left_var];
                                        std::string right_reg = right_var == target_var ? current_var : vars[right_var];
                                        
                                        if (vars.count(left_var) && vars.count(right_var)) {
                                            // Compute the new value
                                            updated_var = new_tmp("new_val");
                                            emit_qubit_alloc(func, updated_var, QBIT_WIDTH);
                                            
                                            if (op == "+") {
                                                emit_quantum_adder(func, updated_var, left_reg, right_reg, QBIT_WIDTH);
                                            }
                                            else if (op == "-") {
                                                emit_quantum_subtractor(func, updated_var, left_reg, right_reg, QBIT_WIDTH);
                                            }
                                            // Add more operations as needed
                                            
                                            // Now we need to conditionally update the value based on cond_reg
                                            std::string final_val = new_tmp(target_var + "_new");
                                            emit_qubit_alloc(func, final_val, QBIT_WIDTH);
                                            
                                            // Start with the current value
                                            for (int j = 0; j < QBIT_WIDTH; j++) {
                                                func.ops.push_back({
                                                    QOpKind::Custom, "",
                                                    current_var + "[" + std::to_string(j) + "]",
                                                    final_val + "[" + std::to_string(j) + "]",
                                                    0, "q.cx"
                                                });
                                            }
                                            
                                            // Now conditionally apply the update
                                            for (int j = 0; j < QBIT_WIDTH; j++) {
                                                // Calculate delta between new and current value
                                                std::string delta = new_tmp("delta");
                                                emit_qubit_alloc(func, delta, 1);
                                                
                                                func.ops.push_back({
                                                    QOpKind::Custom, "",
                                                    updated_var + "[" + std::to_string(j) + "]",
                                                    delta + "[0]",
                                                    0, "q.cx"
                                                });
                                                
                                                func.ops.push_back({
                                                    QOpKind::Custom, "",
                                                    final_val + "[" + std::to_string(j) + "]",
                                                    delta + "[0]",
                                                    0, "q.cx"
                                                });
                                                
                                                // Apply delta controlled by condition bit
                                                func.ops.push_back({
                                                    QOpKind::Custom, "",
                                                    cond_reg + "[0]",
                                                    delta + "[0]",
                                                    0, "q.cx"
                                                });
                                                
                                                func.ops.push_back({
                                                    QOpKind::Custom, "",
                                                    delta + "[0]",
                                                    final_val + "[" + std::to_string(j) + "]",
                                                    0, "q.cx"
                                                });
                                            }
                                            
                                            // Update the current variable for next iteration
                                            current_var = final_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // After unrolling, update the variable in the main context
            vars[target_var] = current_var;
        }
    }
    else {
        // CLASSICAL MODE - Standard control flow (unchanged)
        // Generate labels
        std::string cond_label = new_tmp("while_cond");
        std::string body_label = new_tmp("while_body");
        std::string exit_label = new_tmp("while_exit");

        // Unconditional jump to condition check
        func.ops.push_back({QOpKind::Jump, "", cond_label});

        // Label for the loop body
        func.ops.push_back({QOpKind::Label, body_label});

        // Process body
        process_node(body);

        // After body, jump back to condition
        func.ops.push_back({QOpKind::Jump, "", cond_label});

        // Label for the condition
        func.ops.push_back({QOpKind::Label, cond_label});

        // Evaluate condition and store result in a temporary
        std::string cond_tmp = new_tmp("cond");
        if (condition["kind"] == "BinaryOperator") {
            std::string op = condition.value("opcode", "");
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

            if (condition.contains("inner") && condition["inner"].size() >= 2 &&
                find_ref(condition["inner"][0], left_var) &&
                find_ref(condition["inner"][1], right_var) &&
                vars.count(left_var) && vars.count(right_var)) {
                
                // For different operators we need different comparisons
                if (op == "<") {
                    func.ops.push_back({QOpKind::Sub, cond_tmp, vars[right_var], vars[left_var]});
                } else if (op == ">") {
                    func.ops.push_back({QOpKind::Sub, cond_tmp, vars[left_var], vars[right_var]});
                } else if (op == "==") {
                    // For equality, we subtract and check if the result is 0
                    std::string diff = new_tmp("diff");
                    func.ops.push_back({QOpKind::Sub, diff, vars[left_var], vars[right_var]});
                    // Then convert to a boolean based on whether diff == 0
                    // This is a placeholder for proper equality checking
                    func.ops.push_back({QOpKind::Const, cond_tmp, "", "", 1});
                }
                // Add more comparisons as needed
            }
        }

        // Conditional branch based on cond_tmp
        func.ops.push_back({QOpKind::CBranch, cond_tmp, body_label, exit_label});

        // Label for exit
        func.ops.push_back({QOpKind::Label, exit_label});
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
                            else if (op == "||")
                                emit_quantum_or(func, result, vars[left_var], vars[right_var], QBIT_WIDTH);
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
                            else if (op == "~") {  
                                if (quantum_mode)
                                    emit_quantum_not(func, tmp, vars[var], QBIT_WIDTH);
                                else
                                    func.ops.push_back({QOpKind::Not, tmp, vars[var]});
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
        else if (kind == "WhileStmt") process_while_stmt(node);
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

