#pragma once
#include <string>
#include <vector>
#include <ostream>

enum class QOpKind { Const, Add, Sub, Mul, Print, Return };

struct QMLIR_Op {
    QOpKind kind;
    std::string result;
    std::string lhs, rhs;
    int constant = 0;

    void emit(std::ostream& os) const {
        switch (kind) {
            case QOpKind::Const:
                os << "%" << result << " = q.const " << constant << " : i32\n";
                break;
            case QOpKind::Add:
                os << "%" << result << " = q.addi %" << lhs << ", %" << rhs << " : i32\n";
                break;
            case QOpKind::Sub:
                os << "%" << result << " = q.subi %" << lhs << ", %" << rhs << " : i32\n";
                break;
            case QOpKind::Mul:
                os << "%" << result << " = q.muli %" << lhs << ", %" << rhs << " : i32\n";
                break;
            case QOpKind::Print:
                os << "q.print %" << lhs << "\n";
                break;
            case QOpKind::Return:
                os << "return\n";
                break;
        }
    }
};

struct QMLIR_Function {
    std::string name;
    std::vector<QMLIR_Op> ops;

    void emit(std::ostream& os) const {
        os << "func @" << name << "() -> () {\n";
        for (const auto& op : ops) {
            os << "  ";
            op.emit(os);
        }
        os << "}\n";
    }
};

