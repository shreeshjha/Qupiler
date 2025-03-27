#include "qmlir_ir.hpp"
#include "json.hpp"
#define SIMPLIFIED_AST_IMPLEMENTATION
#include "json_to_simplified_ast.cpp" 
#include <unordered_map>
#include <fstream>

int tmp_id = 0;
std::string new_tmp() { return "t" + std::to_string(tmp_id++); }

void gen_ir(const ASTNode* node, QMLIR_Function& func, std::unordered_map<std::string, std::string>& vars) {
    if (auto* decl = dynamic_cast<const VarDecl*>(node)) {
        std::string var = decl->name.value();
        std::string tmp = new_tmp();
        vars[var] = tmp;
        func.ops.push_back({QOpKind::Const, tmp, "", "", 0});
    }

    else if (auto* bin = dynamic_cast<const BinaryOp*>(node)) {
        std::string lhs = vars[dynamic_cast<DeclRefExpr*>(bin->left.get())->ref];
        std::string rhs = vars[dynamic_cast<DeclRefExpr*>(bin->right.get())->ref];
        std::string out = new_tmp();

        QOpKind op;
        if (bin->op == "+") op = QOpKind::Add;
        else if (bin->op == "-") op = QOpKind::Sub;
        else if (bin->op == "*") op = QOpKind::Mul;
        else throw std::runtime_error("Unsupported op");

        func.ops.push_back({op, out, lhs, rhs});
        vars["last_result"] = out;
    }

    else if (auto* call = dynamic_cast<const CallExpr*>(node)) {
        for (const auto& arg : call->args) {
            if (auto* ref = dynamic_cast<DeclRefExpr*>(arg.get())) {
                func.ops.push_back({QOpKind::Print, "", ref->ref});
            }
        }
    }

    else if (auto* block = dynamic_cast<const CompoundStmt*>(node)) {
        for (const auto& stmt : block->body)
            gen_ir(stmt.get(), func, vars);
    }

    else if (auto* ret = dynamic_cast<const ReturnStmt*>(node)) {
        func.ops.push_back({QOpKind::Return});
    }

    else if (auto* fn = dynamic_cast<const FunctionDecl*>(node)) {
        func.name = fn->name.value();
        for (const auto& inner : fn->paramsAndBody)
            gen_ir(inner.get(), func, vars);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_json> <output_mlir>\n";
        return 1;
    } 
    std::ifstream in(argv[1]);
    nlohmann::json j;
    in >> j;

    auto root = from_dict(j);

    QMLIR_Function fn;
    std::unordered_map<std::string, std::string> vars;
    
    if (auto* tu = dynamic_cast<TranslationUnitDecl*>(root.get())) {
        for (const auto& decl : tu->decls) {
            gen_ir(decl.get(), fn, vars);
        }
    }

    std::ofstream out(argv[2]);
    fn.emit(out);

    return 0;
}

