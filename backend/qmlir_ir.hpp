#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

enum class QOpKind {
    Const,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Print,
    Return,
    Custom  // For quantum-specific operations.
};

struct QMLIR_Op {
    QOpKind kind;
    std::string result;    // Destination register (if applicable)
    std::string lhs;       // Left operand
    std::string rhs;       // Right operand
    int imm = 0;           // Immediate value (if applicable)

    // Custom operation info
    std::string custom_op_name;
    int custom_arg = 0;    // For example, register size for q.alloc

    void emit(std::ostream& out) const {
        std::ostringstream oss;
        switch (kind) {
            case QOpKind::Const:
                oss << "  %" << result << " = q.const " << imm << " : i32";
                break;
            case QOpKind::Add:
                oss << "  %" << result << " = q.addi %" << lhs << ", %" << rhs << " : i32";
                break;
            case QOpKind::Sub:
                oss << "  %" << result << " = q.subi %" << lhs << ", %" << rhs << " : i32";
                break;
            case QOpKind::Mul:
                oss << "  %" << result << " = q.muli %" << lhs << ", %" << rhs << " : i32";
                break;
            case QOpKind::Div:
                oss << "  %" << result << " = q.divi %" << lhs << ", %" << rhs << " : i32";
                break;
            case QOpKind::Mod:
                oss << "  %" << result << " = q.modi %" << lhs << ", %" << rhs << " : i32";
                break;
            case QOpKind::Print:
                oss << "  q.print %" << lhs;
                break;
            case QOpKind::Return:
                oss << "  return";
                break;
            case QOpKind::Custom:
                if (custom_op_name == "q.alloc") {
                    oss << "  %" << result << " = q.alloc : !qreg<" << custom_arg << ">";
                } else if (custom_op_name == "q.init") {
                    oss << "  q.init %" << result << ", " << imm << " : i32";
                } else if (custom_op_name == "q.measure") {
                    oss << "  %" << result << " = q.measure %" << lhs << " : !qreg -> i32";
                } else if (custom_op_name == "q.cx") {
                    // q.cx gate: two operands (lhs: control, rhs: target)
                    oss << "  q.cx %" << lhs << ", %" << rhs;
                } else if (custom_op_name == "q.x") {
                    // q.x gate: one operand
                    oss << "  q.x %" << lhs;
                } else if (custom_op_name == "q.ccx") {
                    // q.ccx gate: three operands (lhs, rhs, and third operand in result)
                    oss << "  q.ccx %" << lhs << ", %" << rhs << ", %" << result;
                } else {
                    // Fallback: if a result is provided, assume assignment
                    if (!result.empty()) {
                        oss << "  %" << result << " = " << custom_op_name << " %" << lhs << ", %" << rhs;
                    } else {
                        oss << "  " << custom_op_name << " %" << lhs << ", %" << rhs;
                    }
                }
                break;
        }
        out << oss.str() << std::endl;
    }
};

struct QMLIR_Function {
    std::string name = "main";
    std::vector<QMLIR_Op> ops;

    void emit(std::ostream& out) const {
        out << "func @" << name << "() -> () {\n";
        for (const auto& op : ops) {
            op.emit(out);
        }
        out << "}\n";
    }
};

