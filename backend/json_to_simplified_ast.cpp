#include <iostream>
#include <stdio.h>
#include <string>
#include <optional>
#include <variant>
#include <vector>
#include <fstream>
#include <stdexcept>
#include "./json.hpp"

using json = nlohmann::json;

class ASTNode {
public: 
    std::string kind;
    std::optional<std::string> name = std::nullopt;
    std::optional<int> addr = std::nullopt;

    // This is completely optional and may not be required
    ASTNode(const std::string& k, const std::optional<std::string>& n = std::nullopt, const std::optional<int> &a = std::nullopt) : kind(k), name(n), addr(a) {}
    
    virtual json to_dict() const {
      json j;
      j["kind"] = kind;
      if (name) j["name"] = *name;
      if (addr) j["addr"] = *addr;
      return j;
    }


    virtual ~ASTNode() = default;

};


class ListNode : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> list;

    ListNode(const std::string& k,
             std::vector<std::unique_ptr<ASTNode>> lst,
             const std::optional<std::string>& n = std::nullopt,
             const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          list(std::move(lst)) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["list"] = json::array();
      for (const auto& item : list)
        j["list"].push_back(item->to_dict());
      return j;
    }

};

class BinaryOp : public ASTNode {
public:
    std::string type;
    std::string op;
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;

    BinaryOp(const std::string& k,
             const std::string& t,
             const std::string& o,
             std::unique_ptr<ASTNode> l,
             std::unique_ptr<ASTNode> r,
             const std::optional<std::string>& n = std::nullopt,
             const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          type(t),
          op(o),
          left(std::move(l)),
          right(std::move(r)) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["type"] = type;
      j["op"] = op;
      j["left"] = left->to_dict();
      j["right"] = right->to_dict();
      return j;
    }

};
class TranslationUnitDecl : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> decls;

    TranslationUnitDecl(const std::string& k,
                        std::vector<std::unique_ptr<ASTNode>> d,
                        const std::optional<std::string>& n = std::nullopt,
                        const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), decls(std::move(d)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["decls"] = json::array();
        for (const auto& d : decls)
            j["decls"].push_back(d->to_dict());
        return j;
    }
};

class FunctionDecl : public ASTNode {
public:
    std::string returnType;
    std::vector<std::unique_ptr<ASTNode>> paramsAndBody;

    FunctionDecl(const std::string& k,
                 const std::string& retType,
                 std::vector<std::unique_ptr<ASTNode>> inner,
                 const std::optional<std::string>& n = std::nullopt,
                 const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), returnType(retType), paramsAndBody(std::move(inner)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["returnType"] = returnType;
        j["inner"] = json::array();
        for (const auto& node : paramsAndBody)
            j["inner"].push_back(node->to_dict());
        return j;
    }
};

class ParmVarDecl : public ASTNode {
public:
    std::string type;

    ParmVarDecl(const std::string& k,
                const std::string& t,
                const std::optional<std::string>& n = std::nullopt,
                const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), type(t) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["type"] = type;
        return j;
    }
};

class VarDecl : public ASTNode {
public:
    std::string type;

    VarDecl(const std::string& k,
            const std::string& t,
            const std::optional<std::string>& n = std::nullopt,
            const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), type(t) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["type"] = type;
        return j;
    }
};

class CompoundStmt : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> body;

    CompoundStmt(const std::string& k,
                 std::vector<std::unique_ptr<ASTNode>> b,
                 const std::optional<std::string>& n = std::nullopt,
                 const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), body(std::move(b)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["body"] = json::array();
        for (const auto& stmt : body)
            j["body"].push_back(stmt->to_dict());
        return j;
    }
};

class ReturnStmt : public ASTNode {
public:
    std::unique_ptr<ASTNode> value;

    ReturnStmt(const std::string& k,
               std::unique_ptr<ASTNode> v,
               const std::optional<std::string>& n = std::nullopt,
               const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), value(std::move(v)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["value"] = value->to_dict();
        return j;
    }
};

class DeclStmt : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> decls;

    DeclStmt(const std::string& k,
             std::vector<std::unique_ptr<ASTNode>> d,
             const std::optional<std::string>& n = std::nullopt,
             const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), decls(std::move(d)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["decls"] = json::array();
        for (const auto& d : decls)
            j["decls"].push_back(d->to_dict());
        return j;
    }
};

class IfStmt : public ASTNode {
public:
    std::unique_ptr<ASTNode> cond;
    std::unique_ptr<ASTNode> thenBranch;
    std::unique_ptr<ASTNode> elseBranch;

    IfStmt(const std::string& k,
           std::unique_ptr<ASTNode> c,
           std::unique_ptr<ASTNode> t,
           std::unique_ptr<ASTNode> e = nullptr,
           const std::optional<std::string>& n = std::nullopt,
           const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), cond(std::move(c)), thenBranch(std::move(t)), elseBranch(std::move(e)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["cond"] = cond->to_dict();
        j["then"] = thenBranch->to_dict();
        if (elseBranch) j["else"] = elseBranch->to_dict();
        return j;
    }
};

class CallExpr : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> args;

    CallExpr(const std::string& k,
             std::vector<std::unique_ptr<ASTNode>> a,
             const std::optional<std::string>& n = std::nullopt,
             const std::optional<int>& a_ = std::nullopt)
        : ASTNode(k, n, a_), args(std::move(a)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["args"] = json::array();
        for (const auto& arg : args)
            j["args"].push_back(arg->to_dict());
        return j;
    }
};

class DeclRefExpr : public ASTNode {
public:
    std::string ref;

    DeclRefExpr(const std::string& k,
                const std::string& r,
                const std::optional<std::string>& n = std::nullopt,
                const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), ref(r) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["ref"] = ref;
        return j;
    }
};

class ForStmt : public ASTNode {
public:
    std::unique_ptr<ASTNode> init, cond, inc, body;

    ForStmt(const std::string& k,
            std::unique_ptr<ASTNode> i,
            std::unique_ptr<ASTNode> c,
            std::unique_ptr<ASTNode> in,
            std::unique_ptr<ASTNode> b,
            const std::optional<std::string>& n = std::nullopt,
            const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          init(std::move(i)), cond(std::move(c)), inc(std::move(in)), body(std::move(b)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["init"] = init->to_dict();
        j["cond"] = cond->to_dict();
        j["inc"] = inc->to_dict();
        j["body"] = body->to_dict();
        return j;
    }
};

class WhileStmt : public ASTNode {
public:
    std::unique_ptr<ASTNode> cond;
    std::unique_ptr<ASTNode> body;

    WhileStmt(const std::string& k,
              std::unique_ptr<ASTNode> c,
              std::unique_ptr<ASTNode> b,
              const std::optional<std::string>& n = std::nullopt,
              const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), cond(std::move(c)), body(std::move(b)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["cond"] = cond->to_dict();
        j["body"] = body->to_dict();
        return j;
    }
};

class DoStmt : public ASTNode {
public:
    std::unique_ptr<ASTNode> body;
    std::unique_ptr<ASTNode> cond;

    DoStmt(const std::string& k,
           std::unique_ptr<ASTNode> b,
           std::unique_ptr<ASTNode> c,
           const std::optional<std::string>& n = std::nullopt,
           const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), body(std::move(b)), cond(std::move(c)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["body"] = body->to_dict();
        j["cond"] = cond->to_dict();
        return j;
    }
};

class BreakStmt : public ASTNode {
public:
    BreakStmt(const std::string& k,
              const std::optional<std::string>& n = std::nullopt,
              const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a) {}

    json to_dict() const override {
        return ASTNode::to_dict();
    }
};

class ContinueStmt : public ASTNode {
public:
    ContinueStmt(const std::string& k,
                 const std::optional<std::string>& n = std::nullopt,
                 const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a) {}

    json to_dict() const override {
        return ASTNode::to_dict();
    }
};

class ImplicitCastExpr : public ASTNode {
public:
    std::string castType;
    std::unique_ptr<ASTNode> subExpr;

    ImplicitCastExpr(const std::string& k,
                     const std::string& t,
                     std::unique_ptr<ASTNode> s,
                     const std::optional<std::string>& n = std::nullopt,
                     const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), castType(t), subExpr(std::move(s)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["castType"] = castType;
        j["subExpr"] = subExpr->to_dict();
        return j;
    }
};


class ParenExpr : public ASTNode {
public:
    std::unique_ptr<ASTNode> expr;

    ParenExpr(const std::string& k,
              std::unique_ptr<ASTNode> e,
              const std::optional<std::string>& n = std::nullopt,
              const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), expr(std::move(e)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["expr"] = expr->to_dict();
        return j;
    }
};

class StringLiteral : public ASTNode {
public:
    std::string value;

    StringLiteral(const std::string& k,
                  const std::string& v,
                  const std::optional<std::string>& n = std::nullopt,
                  const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), value(v) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["value"] = value;
        return j;
    }
};

class ArraySubscriptExpr : public ASTNode {
public:
    std::unique_ptr<ASTNode> base;
    std::unique_ptr<ASTNode> index;

    ArraySubscriptExpr(const std::string& k,
                       std::unique_ptr<ASTNode> b,
                       std::unique_ptr<ASTNode> i,
                       const std::optional<std::string>& n = std::nullopt,
                       const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), base(std::move(b)), index(std::move(i)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["base"] = base->to_dict();
        j["index"] = index->to_dict();
        return j;
    }
};


class CharacterLiteral : public ASTNode {
public:
    std::string value;

    CharacterLiteral(const std::string& k,
                     const std::string& v,
                     const std::optional<std::string>& n = std::nullopt,
                     const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), value(v) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["value"] = value;
        return j;
    }
};


class FloatingLiteral : public ASTNode {
public:
    std::string value;

    FloatingLiteral(const std::string& k,
                    const std::string& v,
                    const std::optional<std::string>& n = std::nullopt,
                    const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), value(v) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["value"] = value;
        return j;
    }
};


class ClangUnaryOperator : public ASTNode {
public:
    std::string op;
    std::unique_ptr<ASTNode> operand;

    ClangUnaryOperator(const std::string& k,
                       const std::string& o,
                       std::unique_ptr<ASTNode> oper,
                       const std::optional<std::string>& n = std::nullopt,
                       const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), op(o), operand(std::move(oper)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["op"] = op;
        j["operand"] = operand->to_dict();
        return j;
    }
};


class CastExpr : public ASTNode {
public:
    std::string castKind;
    std::unique_ptr<ASTNode> subExpr;

    CastExpr(const std::string& k,
             const std::string& ck,
             std::unique_ptr<ASTNode> sub,
             const std::optional<std::string>& n = std::nullopt,
             const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), castKind(ck), subExpr(std::move(sub)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["castKind"] = castKind;
        j["subExpr"] = subExpr->to_dict();
        return j;
    }
};


class InitListExpr : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> inits;

    InitListExpr(const std::string& k,
                 std::vector<std::unique_ptr<ASTNode>> i,
                 const std::optional<std::string>& n = std::nullopt,
                 const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), inits(std::move(i)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["inits"] = json::array();
        for (const auto& i : inits)
            j["inits"].push_back(i->to_dict());
        return j;
    }
};


class CompoundAssignOperator : public ASTNode {
public:
    std::string op;
    std::unique_ptr<ASTNode> lhs;
    std::unique_ptr<ASTNode> rhs;

    CompoundAssignOperator(const std::string& k,
                           const std::string& o,
                           std::unique_ptr<ASTNode> l,
                           std::unique_ptr<ASTNode> r,
                           const std::optional<std::string>& n = std::nullopt,
                           const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), op(o), lhs(std::move(l)), rhs(std::move(r)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["op"] = op;
        j["lhs"] = lhs->to_dict();
        j["rhs"] = rhs->to_dict();
        return j;
    }
};


class NullStmt : public ASTNode {
public:
    NullStmt(const std::string& k,
             const std::optional<std::string>& n = std::nullopt,
             const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a) {}

    json to_dict() const override {
        return ASTNode::to_dict();
    }
};


class LabelStmt : public ASTNode {
public:
    std::string label;
    std::unique_ptr<ASTNode> sub;

    LabelStmt(const std::string& k,
              const std::string& l,
              std::unique_ptr<ASTNode> s,
              const std::optional<std::string>& n = std::nullopt,
              const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), label(l), sub(std::move(s)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["label"] = label;
        j["sub"] = sub->to_dict();
        return j;
    }
};


class GotoStmt : public ASTNode {
public:
    std::string label;

    GotoStmt(const std::string& k,
             const std::string& l,
             const std::optional<std::string>& n = std::nullopt,
             const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), label(l) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["label"] = label;
        return j;
    }
};

class ConditionalOperator : public ASTNode {
public:
    std::unique_ptr<ASTNode> cond;
    std::unique_ptr<ASTNode> trueExpr;
    std::unique_ptr<ASTNode> falseExpr;

    ConditionalOperator(const std::string& k,
                        std::unique_ptr<ASTNode> c,
                        std::unique_ptr<ASTNode> t,
                        std::unique_ptr<ASTNode> f,
                        const std::optional<std::string>& n = std::nullopt,
                        const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), cond(std::move(c)), trueExpr(std::move(t)), falseExpr(std::move(f)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["cond"] = cond->to_dict();
        j["trueExpr"] = trueExpr->to_dict();
        j["falseExpr"] = falseExpr->to_dict();
        return j;
    }
};

class SwitchStmt : public ASTNode {
public:
    std::unique_ptr<ASTNode> cond;
    std::vector<std::unique_ptr<ASTNode>> body;

    SwitchStmt(const std::string& k,
               std::unique_ptr<ASTNode> c,
               std::vector<std::unique_ptr<ASTNode>> b,
               const std::optional<std::string>& n = std::nullopt,
               const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), cond(std::move(c)), body(std::move(b)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["cond"] = cond->to_dict();
        j["body"] = json::array();
        for (const auto& stmt : body)
            j["body"].push_back(stmt->to_dict());
        return j;
    }
};

class CaseStmt : public ASTNode {
public:
    std::unique_ptr<ASTNode> value;
    std::unique_ptr<ASTNode> sub;

    CaseStmt(const std::string& k,
             std::unique_ptr<ASTNode> v,
             std::unique_ptr<ASTNode> s,
             const std::optional<std::string>& n = std::nullopt,
             const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), value(std::move(v)), sub(std::move(s)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["value"] = value->to_dict();
        j["sub"] = sub->to_dict();
        return j;
    }
};

class DefaultStmt : public ASTNode {
public:
    std::unique_ptr<ASTNode> sub;

    DefaultStmt(const std::string& k,
                std::unique_ptr<ASTNode> s,
                const std::optional<std::string>& n = std::nullopt,
                const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), sub(std::move(s)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["sub"] = sub->to_dict();
        return j;
    }
};


class ReturnVoidStmt : public ASTNode {
public:
    ReturnVoidStmt(const std::string& k,
                   const std::optional<std::string>& n = std::nullopt,
                   const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a) {}

    json to_dict() const override {
        return ASTNode::to_dict();
    }
};

class ContinueWithLabelStmt : public ASTNode {
public:
    std::string label;

    ContinueWithLabelStmt(const std::string& k,
                          const std::string& l,
                          const std::optional<std::string>& n = std::nullopt,
                          const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), label(l) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["label"] = label;
        return j;
    }
};

class TypedefDecl : public ASTNode {
public:
    std::string underlyingType;

    TypedefDecl(const std::string& k,
                const std::string& ut,
                const std::optional<std::string>& n = std::nullopt,
                const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), underlyingType(ut) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["underlyingType"] = underlyingType;
        return j;
    }
};

class EnumDecl : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> enumerators;

    EnumDecl(const std::string& k,
             std::vector<std::unique_ptr<ASTNode>> e,
             const std::optional<std::string>& n = std::nullopt,
             const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), enumerators(std::move(e)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["enumerators"] = json::array();
        for (const auto& e : enumerators)
            j["enumerators"].push_back(e->to_dict());
        return j;
    }
};

class EnumConstantDecl : public ASTNode {
public:
    std::string value;

    EnumConstantDecl(const std::string& k,
                     const std::string& v,
                     const std::optional<std::string>& n = std::nullopt,
                     const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), value(v) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["value"] = value;
        return j;
    }
};

class PredefinedExpr : public ASTNode {
public:
    std::string text;

    PredefinedExpr(const std::string& k,
                   const std::string& t,
                   const std::optional<std::string>& n = std::nullopt,
                   const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), text(t) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["text"] = text;
        return j;
    }
};

class CStyleCastExpr : public ASTNode {
public:
    std::string castType;
    std::unique_ptr<ASTNode> subExpr;

    CStyleCastExpr(const std::string& k,
                   const std::string& t,
                   std::unique_ptr<ASTNode> s,
                   const std::optional<std::string>& n = std::nullopt,
                   const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), castType(t), subExpr(std::move(s)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["castType"] = castType;
        j["subExpr"] = subExpr->to_dict();
        return j;
    }
};


class NullPointerLiteralExpr : public ASTNode {
public:
    std::string type;

    NullPointerLiteralExpr(const std::string& k,
                           const std::string& t,
                           const std::optional<std::string>& n = std::nullopt,
                           const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), type(t) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["type"] = type;
        return j;
    }
};

class CompoundLiteralExpr : public ASTNode {
public:
    std::string type;
    std::unique_ptr<ASTNode> initializer;

    CompoundLiteralExpr(const std::string& k,
                        const std::string& t,
                        std::unique_ptr<ASTNode> i,
                        const std::optional<std::string>& n = std::nullopt,
                        const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), type(t), initializer(std::move(i)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["type"] = type;
        j["initializer"] = initializer->to_dict();
        return j;
    }
};

class DesignatedInitExpr : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> initList;

    DesignatedInitExpr(const std::string& k,
                       std::vector<std::unique_ptr<ASTNode>> inits,
                       const std::optional<std::string>& n = std::nullopt,
                       const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), initList(std::move(inits)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["inits"] = json::array();
        for (const auto& i : initList)
            j["inits"].push_back(i->to_dict());
        return j;
    }
};

class GenericSelectionExpr : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> inner;

    GenericSelectionExpr(const std::string& k,
                         std::vector<std::unique_ptr<ASTNode>> i,
                         const std::optional<std::string>& n = std::nullopt,
                         const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), inner(std::move(i)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["inner"] = json::array();
        for (const auto& node : inner)
            j["inner"].push_back(node->to_dict());
        return j;
    }
};

class StaticAssertDecl : public ASTNode {
public:
    std::unique_ptr<ASTNode> condition;
    std::string message;

    StaticAssertDecl(const std::string& k,
                     std::unique_ptr<ASTNode> cond,
                     const std::string& msg,
                     const std::optional<std::string>& n = std::nullopt,
                     const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), condition(std::move(cond)), message(msg) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["condition"] = condition->to_dict();
        j["message"] = message;
        return j;
    }
};

class OffsetOfExpr : public ASTNode {
public:
    std::string type;

    OffsetOfExpr(const std::string& k,
                 const std::string& t,
                 const std::optional<std::string>& n = std::nullopt,
                 const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), type(t) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["type"] = type;
        return j;
    }
};

class SizeOfPackExpr : public ASTNode {
public:
    std::string identifier;

    SizeOfPackExpr(const std::string& k,
                   const std::string& id,
                   const std::optional<std::string>& n = std::nullopt,
                   const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), identifier(id) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["identifier"] = identifier;
        return j;
    }
};

class VAArgExpr : public ASTNode {
public:
    std::string type;
    std::unique_ptr<ASTNode> expr;

    VAArgExpr(const std::string& k,
              const std::string& t,
              std::unique_ptr<ASTNode> e,
              const std::optional<std::string>& n = std::nullopt,
              const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), type(t), expr(std::move(e)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["type"] = type;
        j["expr"] = expr->to_dict();
        return j;
    }
};

class TypeTraitExpr : public ASTNode {
public:
    std::string trait;

    TypeTraitExpr(const std::string& k,
                  const std::string& t,
                  const std::optional<std::string>& n = std::nullopt,
                  const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), trait(t) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["trait"] = trait;
        return j;
    }
};

class AlignOfExpr : public ASTNode {
public:
    std::string type;

    AlignOfExpr(const std::string& k,
                const std::string& t,
                const std::optional<std::string>& n = std::nullopt,
                const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), type(t) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["type"] = type;
        return j;
    }
};

class UnaryExprOrTypeTraitExpr : public ASTNode {
public:
    std::string exprType;

    UnaryExprOrTypeTraitExpr(const std::string& k,
                              const std::string& et,
                              const std::optional<std::string>& n = std::nullopt,
                              const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), exprType(et) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["exprType"] = exprType;
        return j;
    }
};

class MemberExpr : public ASTNode {
public:
    std::string memberName;
    std::unique_ptr<ASTNode> base;

    MemberExpr(const std::string& k,
               const std::string& m,
               std::unique_ptr<ASTNode> b,
               const std::optional<std::string>& n = std::nullopt,
               const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), memberName(m), base(std::move(b)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["member"] = memberName;
        j["base"] = base->to_dict();
        return j;
    }
};

class Designator : public ASTNode {
public:
    std::string field;

    Designator(const std::string& k,
               const std::string& f,
               const std::optional<std::string>& n = std::nullopt,
               const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), field(f) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["field"] = field;
        return j;
    }
};

class AsmStmt : public ASTNode {
public:
    std::string asmString;

    AsmStmt(const std::string& k,
            const std::string& s,
            const std::optional<std::string>& n = std::nullopt,
            const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), asmString(s) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["asm"] = asmString;
        return j;
    }
};

class ImaginaryLiteral : public ASTNode {
public:
    std::unique_ptr<ASTNode> value;

    ImaginaryLiteral(const std::string& k,
                     std::unique_ptr<ASTNode> v,
                     const std::optional<std::string>& n = std::nullopt,
                     const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), value(std::move(v)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["value"] = value->to_dict();
        return j;
    }
};

class OMPParallelDirective : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> body;

    OMPParallelDirective(const std::string& k,
                         std::vector<std::unique_ptr<ASTNode>> b,
                         const std::optional<std::string>& n = std::nullopt,
                         const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), body(std::move(b)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["body"] = json::array();
        for (const auto& s : body)
            j["body"].push_back(s->to_dict());
        return j;
    }
};

class OMPForDirective : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> body;

    OMPForDirective(const std::string& k,
                    std::vector<std::unique_ptr<ASTNode>> b,
                    const std::optional<std::string>& n = std::nullopt,
                    const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), body(std::move(b)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["body"] = json::array();
        for (const auto& s : body)
            j["body"].push_back(s->to_dict());
        return j;
    }
};

class FunctionProtoType : public ASTNode {
public:
    std::string type;

    FunctionProtoType(const std::string& k,
                      const std::string& t,
                      const std::optional<std::string>& n = std::nullopt,
                      const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), type(t) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["type"] = type;
        return j;
    }
};

class PointerType : public ASTNode {
public:
    std::string pointee;

    PointerType(const std::string& k,
                const std::string& p,
                const std::optional<std::string>& n = std::nullopt,
                const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), pointee(p) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["pointee"] = pointee;
        return j;
    }
};

class RecordDecl : public ASTNode {
public:
    std::string recordType;  // e.g., "struct" or "union"
    std::vector<std::unique_ptr<ASTNode>> fields;

    RecordDecl(const std::string& k,
               const std::string& rt,
               std::vector<std::unique_ptr<ASTNode>> f,
               const std::optional<std::string>& n = std::nullopt,
               const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), recordType(rt), fields(std::move(f)) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["recordType"] = recordType;
        j["fields"] = json::array();
        for (const auto& f : fields)
            j["fields"].push_back(f->to_dict());
        return j;
    }
};

class FieldDecl : public ASTNode {
public:
    std::string type;

    FieldDecl(const std::string& k,
              const std::string& t,
              const std::optional<std::string>& n = std::nullopt,
              const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), type(t) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["type"] = type;
        return j;
    }
};

class AvailabilityAttr : public ASTNode {
public:
    std::string platform;

    AvailabilityAttr(const std::string& k,
                     const std::string& p,
                     const std::optional<std::string>& n = std::nullopt,
                     const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), platform(p) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["platform"] = platform;
        return j;
    }
};

class BuiltinAttr : public ASTNode {
public:
    std::string builtin;

    BuiltinAttr(const std::string& k,
                const std::string& b,
                const std::optional<std::string>& n = std::nullopt,
                const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), builtin(b) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["builtin"] = builtin;
        return j;
    }
};

class FormatAttr : public ASTNode {
public:
    std::string formatKind;

    FormatAttr(const std::string& k,
               const std::string& f,
               const std::optional<std::string>& n = std::nullopt,
               const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), formatKind(f) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["formatKind"] = formatKind;
        return j;
    }
};

class AsmLabelAttr : public ASTNode {
public:
    std::string label;

    AsmLabelAttr(const std::string& k,
                 const std::string& lbl,
                 const std::optional<std::string>& n = std::nullopt,
                 const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), label(lbl) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["label"] = label;
        return j;
    }
};

class GenericAttr : public ASTNode {
public:
    json contents;

    GenericAttr(const std::string& k,
                const json& c,
                const std::optional<std::string>& n = std::nullopt,
                const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), contents(c) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        for (auto& [key, val] : contents.items()) {
            if (key != "kind" && key != "name" && key != "addr") {
                j[key] = val;
            }
        }
        return j;
    }
};


class DeprecatedAttr : public ASTNode {
public:
    std::string message;

    DeprecatedAttr(const std::string& k,
                   const std::string& msg,
                   const std::optional<std::string>& n = std::nullopt,
                   const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), message(msg) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["message"] = message;
        return j;
    }
};

class IntegerLiteral : public ASTNode {
public:
    std::string value;
    std::string type;

    IntegerLiteral(const std::string& k,
                   const std::string& v,
                   const std::string& t,
                   const std::optional<std::string>& n = std::nullopt,
                   const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), value(v), type(t) {}

    json to_dict() const override {
        json j = ASTNode::to_dict();
        j["value"] = value;
        j["type"] = type;
        return j;
    }
};


// Helper function to convert a json array to a vector of ASTNode pointers
std::vector<std::unique_ptr<ASTNode>> from_dict_list(const json& arr);
// Main conversion function

std::unique_ptr<ASTNode> from_dict(const json& data) {
    if(data.is_array()) {
        throw std::runtime_error("Expected object but got array");
    }

    if(!data.is_object()) {
        return nullptr;
    }

    // Safely get a string value from JSON, handling cases where it might be an object
    auto safe_get_string = [](const json& j, const std::string& key, const std::string& default_val) -> std::string {
        if (!j.contains(key)) {
            return default_val;
        }
        const auto& value = j[key];
        if (value.is_string()) {
            return value.get<std::string>();
        }
        // If it's not a string, return the default
        return default_val;
    };

    std::string kind = safe_get_string(data, "kind", "");
   
    std::optional<std::string> name = (data.contains("name") 
        && !data["name"].is_null() 
        && data["name"].is_string()) ? std::make_optional(data["name"].get<std::string>()) : std::nullopt;
    
    std::optional<int> addr = (data.contains("addr") && !data["addr"].is_null() && data["addr"].is_number()) 
        ? std::make_optional(data["addr"].get<int>()) : std::nullopt;

      
    if (kind == "List"){
      auto listNodes=from_dict_list(data.at("list"));
      return std::make_unique<ListNode>(kind,std::move(listNodes),name,addr);
    }

  
    else if (kind == "TranslationUnitDecl") {
      auto decls = from_dict_list(data.value("inner", json::array()));
      return std::make_unique<TranslationUnitDecl>(kind, std::move(decls), name, addr);
    }

    else if (kind == "FunctionDecl") {
     std::string retType = safe_get_string(data, "type", "");
     auto inner = from_dict_list(data.value("inner", json::array()));
      return std::make_unique<FunctionDecl>(kind, retType, std::move(inner), name, addr);
    }
    
    else if (kind == "ParmVarDecl") {
      std::string type = safe_get_string(data, "type", "");
      return std::make_unique<ParmVarDecl>(kind, type, name, addr);
    }

    else if (kind == "VarDecl") {
      std::string type = safe_get_string(data, "type", "");
      return std::make_unique<VarDecl>(kind, type, name, addr);
    }

    else if (kind == "CompoundStmt") {
      auto body = from_dict_list(data.value("inner", json::array()));
      return std::make_unique<CompoundStmt>(kind, std::move(body), name, addr);
    }

    else if (kind == "ReturnStmt") {
      std::unique_ptr<ASTNode> value = nullptr;
      if (data.contains("inner") && !data["inner"].empty())
        value = from_dict(data["inner"][0]);
      return std::make_unique<ReturnStmt>(kind, std::move(value), name, addr);
    }
  
    
    else if (kind == "DeclStmt") {
      auto decls = from_dict_list(data.value("inner", json::array()));
      return std::make_unique<DeclStmt>(kind, std::move(decls), name, addr);
    }

    else if (kind == "IfStmt") {
      auto cond = from_dict(data["inner"][0]);
      auto thenStmt = from_dict(data["inner"][1]);
      std::unique_ptr<ASTNode> elseStmt = nullptr;
      if (data["inner"].size() > 2)
        elseStmt = from_dict(data["inner"][2]);
      return std::make_unique<IfStmt>(kind, std::move(cond), std::move(thenStmt), std::move(elseStmt), name, addr);
    }

    else if (kind == "BinaryOperator") {
      std::string type = safe_get_string(data, "type", "");
      std::string op = safe_get_string(data, "opcode", "unknown");
      auto left = from_dict(data["inner"][0]);
      auto right = from_dict(data["inner"][1]);
      return std::make_unique<BinaryOp>("BinaryOp", type, op, std::move(left), std::move(right), name, addr);
    }
    
    else if (kind == "CallExpr") {
        auto args = from_dict_list(data.value("inner", json::array()));
        return std::make_unique<CallExpr>(kind, std::move(args), name, addr);
    }

    else if (kind == "DeclRefExpr") {
        std::string ref = safe_get_string(data, "referencedDecl", "unknown");
        return std::make_unique<DeclRefExpr>(kind, ref, name, addr);
    }
    
    else if (kind == "ForStmt") {
        auto init = from_dict(data["inner"][0]);
        auto cond = from_dict(data["inner"][1]);
        auto inc = from_dict(data["inner"][2]);
        auto body = from_dict(data["inner"][3]);
        return std::make_unique<ForStmt>(kind, std::move(init), std::move(cond), std::move(inc), std::move(body), name, addr);
    }

    else if (kind == "WhileStmt") {
        auto cond = from_dict(data["inner"][0]);
        auto body = from_dict(data["inner"][1]);
        return std::make_unique<WhileStmt>(kind, std::move(cond), std::move(body), name, addr);
    }

    else if (kind == "DoStmt") {
        auto body = from_dict(data["inner"][0]);
        auto cond = from_dict(data["inner"][1]);
        return std::make_unique<DoStmt>(kind, std::move(body), std::move(cond), name, addr);
    }

    else if (kind == "BreakStmt") {
        return std::make_unique<BreakStmt>(kind, name, addr);
    }

    else if (kind == "ContinueStmt") {
        return std::make_unique<ContinueStmt>(kind, name, addr);
    }

    else if (kind == "ImplicitCastExpr") {
        std::string type = safe_get_string(data, "type", "");
        auto inner = from_dict(data["inner"][0]);
        return std::make_unique<ImplicitCastExpr>(kind, type, std::move(inner), name, addr);
    }

    else if (kind == "ParenExpr") {
        auto inner = from_dict(data["inner"][0]);
        return std::make_unique<ParenExpr>(kind, std::move(inner), name, addr);
    }

    else if (kind == "StringLiteral") {
        std::string value = safe_get_string(data, "value", "");
        return std::make_unique<StringLiteral>(kind, value, name, addr);
    }

    else if (kind == "ArraySubscriptExpr") {
      auto base = from_dict(data["inner"][0]);
      auto index = from_dict(data["inner"][1]);
      return std::make_unique<ArraySubscriptExpr>(kind, std::move(base), std::move(index), name, addr);
    }

    else if (kind == "CharacterLiteral") {
      std::string value = safe_get_string(data, "value", "");
      return std::make_unique<CharacterLiteral>(kind, value, name, addr);
    }

    else if (kind == "FloatingLiteral") {
      std::string value = safe_get_string(data, "value", "");
      return std::make_unique<FloatingLiteral>(kind, value, name, addr);
    }
    
    else if (kind == "UnaryOperator") {
      std::string op = safe_get_string(data, "opcode", "");
      auto operand = from_dict(data["inner"][0]);
      return std::make_unique<ClangUnaryOperator>(kind, op, std::move(operand), name, addr);
    }
  
    else if (kind == "CastExpr") {
      std::string castKind = safe_get_string(data, "castKind", "");
      auto subExpr = from_dict(data["inner"][0]);
      return std::make_unique<CastExpr>(kind, castKind, std::move(subExpr), name, addr);
    }
    
    else if (kind == "InitListExpr") {
      auto inits = from_dict_list(data.value("inner", json::array()));
      return std::make_unique<InitListExpr>(kind, std::move(inits), name, addr);
    }

    else if (kind == "CompoundAssignOperator") {
      std::string op = safe_get_string(data, "opcode", "");
      auto lhs = from_dict(data["inner"][0]);
      auto rhs = from_dict(data["inner"][1]);
      return std::make_unique<CompoundAssignOperator>(kind, op, std::move(lhs), std::move(rhs), name, addr);
    } 

    else if (kind == "NullStmt") {
      return std::make_unique<NullStmt>(kind, name, addr);
    }

    else if (kind == "LabelStmt") {
      std::string label = safe_get_string(data, "name", "");
      auto sub = from_dict(data["inner"][0]);
      return std::make_unique<LabelStmt>(kind, label, std::move(sub), name, addr);
    }

    else if (kind == "GotoStmt") {
      std::string label = safe_get_string(data, "label", "");
      return std::make_unique<GotoStmt>(kind, label, name, addr);
    }

    else if (kind == "ConditionalOperator") {
        auto cond = from_dict(data["inner"][0]);
        auto trueExpr = from_dict(data["inner"][1]);
        auto falseExpr = from_dict(data["inner"][2]);
        return std::make_unique<ConditionalOperator>(kind, std::move(cond), std::move(trueExpr), std::move(falseExpr), name, addr);
    }

    else if (kind == "SwitchStmt") {
        auto cond = from_dict(data["inner"][0]);
        std::vector<std::unique_ptr<ASTNode>> body;
        for (size_t i = 1; i < data["inner"].size(); ++i)
            body.push_back(from_dict(data["inner"][i]));
        return std::make_unique<SwitchStmt>(kind, std::move(cond), std::move(body), name, addr);
    }

    
    else if (kind == "CaseStmt") {
        auto value = from_dict(data["inner"][0]);
        auto sub = from_dict(data["inner"][1]);
        return std::make_unique<CaseStmt>(kind, std::move(value), std::move(sub), name, addr);
    }

    else if (kind == "DefaultStmt") {
        auto sub = from_dict(data["inner"][0]);
        return std::make_unique<DefaultStmt>(kind, std::move(sub), name, addr);
    }
    
    else if (kind == "ReturnVoidStmt") {
        return std::make_unique<ReturnVoidStmt>(kind, name, addr);
    }

    else if (kind == "ContinueWithLabelStmt") {
        std::string label = safe_get_string(data, "label", "");
        return std::make_unique<ContinueWithLabelStmt>(kind, label, name, addr);
    }

    else if (kind == "TypedefDecl") {
        std::string underlyingType = safe_get_string(data, "type", "");
        return std::make_unique<TypedefDecl>(kind, underlyingType, name, addr);
    }
    
    else if (kind == "EnumDecl") {
        auto enumerators = from_dict_list(data.value("inner", json::array()));
        return std::make_unique<EnumDecl>(kind, std::move(enumerators), name, addr);
    }
    
    else if (kind == "EnumConstantDecl") {
        std::string value = safe_get_string(data, "value", "0");
        return std::make_unique<EnumConstantDecl>(kind, value, name, addr);
    }

    else if (kind == "PredefinedExpr") {
        std::string text = safe_get_string(data, "text", "");
        return std::make_unique<PredefinedExpr>(kind, text, name, addr);
    }
    
    else if (kind == "CStyleCastExpr") {
        std::string type = safe_get_string(data, "type", "");
        auto subExpr = from_dict(data["inner"][0]);
        return std::make_unique<CStyleCastExpr>(kind, type, std::move(subExpr), name, addr);
    }

    else if (kind == "NullPointerLiteralExpr") {
        std::string type = safe_get_string(data, "type", "");
        return std::make_unique<NullPointerLiteralExpr>(kind, type, name, addr);
    }

    else if (kind == "CompoundLiteralExpr") {
        std::string type = safe_get_string(data, "type", "");
        auto initializer = from_dict(data["inner"][0]);
        return std::make_unique<CompoundLiteralExpr>(kind, type, std::move(initializer), name, addr);
    }
    
    else if (kind == "DesignatedInitExpr") {
        auto inits = from_dict_list(data.value("inner", json::array()));
        return std::make_unique<DesignatedInitExpr>(kind, std::move(inits), name, addr);
    }

    else if (kind == "GenericSelectionExpr") {
        auto inner = from_dict_list(data.value("inner", json::array()));
        return std::make_unique<GenericSelectionExpr>(kind, std::move(inner), name, addr);
    }

    else if (kind == "StaticAssertDecl") {
        auto cond = from_dict(data["inner"][0]);
        std::string msg = safe_get_string(data, "message", "");
        return std::make_unique<StaticAssertDecl>(kind, std::move(cond), msg, name, addr);
    }

    else if (kind == "OffsetOfExpr") {
        std::string type = safe_get_string(data, "type", "");
        return std::make_unique<OffsetOfExpr>(kind, type, name, addr);
    }
    
    else if (kind == "SizeOfPackExpr") {
        std::string id = safe_get_string(data, "name", "");
        return std::make_unique<SizeOfPackExpr>(kind, id, name, addr);
    }
    
    else if (kind == "VAArgExpr") {
        std::string type = safe_get_string(data, "type", "");
        auto expr = from_dict(data["inner"][0]);
        return std::make_unique<VAArgExpr>(kind, type, std::move(expr), name, addr);
    }
    
    else if (kind == "TypeTraitExpr") {
        std::string trait = safe_get_string(data, "trait", "");
        return std::make_unique<TypeTraitExpr>(kind, trait, name, addr);
    }

    else if (kind == "AlignOfExpr") {
        std::string type = safe_get_string(data, "type", "");
        return std::make_unique<AlignOfExpr>(kind, type, name, addr);
    }

    else if (kind == "UnaryExprOrTypeTraitExpr") {
        std::string exprType = safe_get_string(data, "type", ""); 
        return std::make_unique<UnaryExprOrTypeTraitExpr>(kind, exprType, name, addr);
    }
    
    else if (kind == "MemberExpr") {
        std::string memberName = safe_get_string(data, "name", "");
        auto base = from_dict(data["inner"][0]);
        return std::make_unique<MemberExpr>(kind, memberName, std::move(base), name, addr);
    }
    
    else if (kind == "Designator") {
        std::string field = safe_get_string(data, "name", "");
        return std::make_unique<Designator>(kind, field, name, addr);
    }
    
    else if (kind == "AsmStmt") {
        std::string s = safe_get_string(data, "asmString", "");
        return std::make_unique<AsmStmt>(kind, s, name, addr);
    }
    
    else if (kind == "ImaginaryLiteral") {
        auto value = from_dict(data["inner"][0]);
        return std::make_unique<ImaginaryLiteral>(kind, std::move(value), name, addr);
    }
    
    else if (kind == "OMPParallelDirective") {
        auto body = from_dict_list(data.value("inner", json::array()));
        return std::make_unique<OMPParallelDirective>(kind, std::move(body), name, addr);
    }

    else if (kind == "OMPForDirective") {
        auto body = from_dict_list(data.value("inner", json::array()));
        return std::make_unique<OMPForDirective>(kind, std::move(body), name, addr);
    }
    
    else if (kind == "FunctionProtoType") {
        std::string type = safe_get_string(data, "type", "");
        return std::make_unique<FunctionProtoType>(kind, type, name, addr);
    }
    
    else if (kind == "PointerType") {
        std::string pointee = safe_get_string(data, "type", "");
        return std::make_unique<PointerType>(kind, pointee, name, addr);
    }

    else if (kind == "RecordDecl") {
      std::string recordType = safe_get_string(data, "tagUsed", "struct");
      auto fields = from_dict_list(data.value("inner", json::array()));
      return std::make_unique<RecordDecl>(kind, recordType, std::move(fields), name, addr);
    }

    else if (kind == "FieldDecl") {
      std::string type = safe_get_string(data, "type", "");
      return std::make_unique<FieldDecl>(kind, type, name, addr);
    }
    
    else if (kind == "AvailabilityAttr") {
      std::string platform = safe_get_string(data, "platform", "unknown");
      return std::make_unique<AvailabilityAttr>(kind, platform, name, addr);
    }
  
    else if (kind == "BuiltinAttr") {
      std::string builtin = safe_get_string(data, "builtin", "true");
      return std::make_unique<BuiltinAttr>(kind, builtin, name, addr);
    }
    
    else if (kind == "FormatAttr") {
      std::string fmt = safe_get_string(data, "type", "format");
      return std::make_unique<FormatAttr>(kind, fmt, name, addr);
    }

    else if (kind == "AsmLabelAttr") {
      std::string label = safe_get_string(data, "label", "");
      return std::make_unique<AsmLabelAttr>(kind, label, name, addr);
    }
    
    else if (kind == "DeprecatedAttr") {
      std::string msg = safe_get_string(data, "message", "");
      return std::make_unique<DeprecatedAttr>(kind, msg, name, addr);
    }
    
    else if (kind.size() >= 4 && kind.substr(kind.size() - 4) == "Attr"){
      return std::make_unique<GenericAttr>(kind, data, name, addr);
    }
    
    else if (kind == "IntegerLiteral") {
      std::string value = safe_get_string(data, "value", "0");
      std::string type = safe_get_string(data, "type", "");
      return std::make_unique<IntegerLiteral>(kind, value, type, name, addr);
    }



    else{
      throw std::runtime_error("Unknown kind: " + kind);
    }
}
// Helper that converts a JSON array to a vector of ASTNode pointers
std::vector<std::unique_ptr<ASTNode>> from_dict_list(const json& arr) {
    std::vector<std::unique_ptr<ASTNode>> nodes;
    for (const auto& item : arr) {
        auto node = from_dict(item);
        nodes.push_back(std::move(node));  // Correct: Moving the unique_ptr
    }
    return nodes;
}


#ifndef SIMPLIFIED_AST_IMPLEMENTATION
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_json> <output_json>\n";
        return 1;
    }

    std::ifstream inFile(argv[1]);
    if (!inFile) {
        std::cerr << "Failed to open input file: " << argv[1] << "\n";
        return 1;
    }

    json inputJson;
    inFile >> inputJson;

    try {
        auto ast = from_dict(inputJson);
        json outputJson = ast->to_dict();

        std::ofstream outFile(argv[2]);
        outFile << outputJson.dump(4);  // Pretty print with indent
        std::cout << "AST written to " << argv[2] << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
#endif

