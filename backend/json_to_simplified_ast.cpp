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

class NamedValue : public ASTNode {
public:
    std::string type;
    std::string symbol;
    std::optional<std::string> constant;

    NamedValue(const std::string& k,
               const std::string& t,
               const std::string& sym,
               const std::optional<std::string>& c = std::nullopt,
               const std::optional<std::string>& n = std::nullopt,
               const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          type(t),
          symbol(sym),
          constant(c) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["type"] = type;
      j["symbol"] = symbol;
      if (constant) j["constant"] = *constant;
      return j;
    }

};

class BinaryOp;
class Conversion;

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

class IntegerLiteral : public ASTNode {
public:
    std::string type;
    std::string value;
    std::string constant;

    IntegerLiteral(const std::string& k,
                   const std::string& t,
                   const std::string& v,
                   const std::string& c,
                   const std::optional<std::string>& n = std::nullopt,
                   const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          type(t),
          value(v),
          constant(c) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["type"] = type;
      j["value"] = value;
      j["constant"] = constant;
      return j;
    }

};


class Conversion : public ASTNode {
public:
    std::string type;
    std::unique_ptr<ASTNode> operand;
    std::optional<std::string> constant = std::nullopt;

    Conversion(const std::string& k,
               const std::string& t,
               std::unique_ptr<ASTNode> op,
               const std::optional<std::string>& c = std::nullopt,
               const std::optional<std::string>& n = std::nullopt,
               const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          type(t),
          operand(std::move(op)),
          constant(c) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["type"] = type;
      j["operand"] = operand->to_dict();
      if (constant) j["constant"] = *constant;
      return j; 
    }

};

class Assignment : public ASTNode {
public:
    std::string type;
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;
    bool isNonBlocking;

    Assignment(const std::string& k,
               const std::string& t,
               std::unique_ptr<ASTNode> l,
               std::unique_ptr<ASTNode> r,
               bool nonBlocking,
               const std::optional<std::string>& n = std::nullopt,
               const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          type(t),
          left(std::move(l)),
          right(std::move(r)),
          isNonBlocking(nonBlocking) {}
    
    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["type"] = type;
      j["isNonBlocking"] = isNonBlocking;
      j["left"] = left->to_dict();
      j["right"] = right->to_dict();
      return j;
    } 

};

class UnaryOp : public ASTNode {
public:
    std::string type;
    std::string op;
    std::unique_ptr<ASTNode> operand;

    UnaryOp(const std::string& k,
            const std::string& t,
            const std::string& o,
            std::unique_ptr<ASTNode> oper,
            const std::optional<std::string>& n = std::nullopt,
            const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          type(t),
          op(o),
          operand(std::move(oper)) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["type"] = type;
      j["op"] = op;
      j["operand"] = operand->to_dict();
      return j;
    }

};


class CompilationUnit : public ASTNode {
public:
    CompilationUnit(const std::string& k,
                    const std::optional<std::string>& n = std::nullopt,
                    const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a) {}

    json to_dict() const override {
      return ASTNode::to_dict();
    }

};

class ContinuousAssign : public ASTNode {
public:
    std::unique_ptr<ASTNode> assignment;

    ContinuousAssign(const std::string& k,
                     std::unique_ptr<ASTNode> assign,
                     const std::optional<std::string>& n = std::nullopt,
                     const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          assignment(std::move(assign)) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["assignment"] = assignment->to_dict();
      return j;
    }

};


class Port : public ASTNode {
public:
    std::string type;
    std::string direction;
    std::string internalSymbol;

    Port(const std::string& k,
         const std::string& t,
         const std::string& d,
         const std::string& internalSym,
         const std::optional<std::string>& n = std::nullopt,
         const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          type(t),
          direction(d),
          internalSymbol(internalSym) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["type"] = type;
      j["direction"] = direction;
      j["internalSymbol"] = internalSymbol;
      return j;
    }

};


class PrimitiveInstance : public ASTNode {
public:
    std::string primitiveType;
    std::vector<std::unique_ptr<ASTNode>> ports;

    PrimitiveInstance(const std::string& k,
                      const std::string& pt,
                      std::vector<std::unique_ptr<ASTNode>> p,
                      const std::optional<std::string>& n = std::nullopt,
                      const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          primitiveType(pt),
          ports(std::move(p)) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["primitiveType"] = primitiveType;
      j["ports"] = json::array();
      for (const auto& p : ports)
        j["ports"].push_back(p->to_dict());
      return j;
    }

};


class Variable;
class Variable : public ASTNode {
public:
    std::string type;
    std::string lifetime;

    Variable(const std::string& k,
             const std::string& t,
             const std::string& l,
             const std::optional<std::string>& n = std::nullopt,
             const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          type(t),
          lifetime(l) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["type"] = type;
      j["lifetime"] = lifetime;
      return j;
    }

};


class ExpressionStatement : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> expr;

    ExpressionStatement(const std::string& k,
                        std::vector<std::unique_ptr<ASTNode>> e,
                        const std::optional<std::string>& n = std::nullopt,
                        const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          expr(std::move(e)) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["expr"] = json::array();
      for (const auto& e : expr)
        j["expr"].push_back(e->to_dict());
      return j;
    }

};



class Block : public ASTNode {
public:
    std::string blockKind;
    std::vector<std::unique_ptr<ASTNode>> body;

    Block(const std::string& k,
          const std::string& bk,
          std::vector<std::unique_ptr<ASTNode>> b,
          const std::optional<std::string>& n = std::nullopt,
          const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          blockKind(bk),
          body(std::move(b)) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["blockKind"] = blockKind;
      j["body"] = json::array();
      for (const auto& stmt : body)
        j["body"].push_back(stmt->to_dict());
      return j;
    }

};



class ProceduralBlock;
class ProceduralBlock : public ASTNode {
public:
    std::unique_ptr<ASTNode> body;
    std::string procedureKind;

    ProceduralBlock(const std::string& k,
                    std::unique_ptr<ASTNode> b,
                    const std::string& pk,
                    const std::optional<std::string>& n = std::nullopt,
                    const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          body(std::move(b)),
          procedureKind(pk) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["procedureKind"] = procedureKind;
      j["body"] = body->to_dict();
      return j;
    }

};


class NetType : public ASTNode {
public:
    std::string type;

    NetType(const std::string& k,
            const std::string& t,
            const std::optional<std::string>& n = std::nullopt,
            const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          type(t) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["type"] = type;
      return j;
    }

};



class Net : public ASTNode {
public:
    std::string type;
    std::unique_ptr<ASTNode> netType;

    Net(const std::string& k,
        const std::string& t,
        std::unique_ptr<ASTNode> nt,
        const std::optional<std::string>& n = std::nullopt,
        const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          type(t),
          netType(std::move(nt)) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["type"] = type;
      j["netType"] = netType->to_dict();
      return j;
    }

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


class InstanceBody;

class InstanceBody : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> members;
    std::string definition;

    InstanceBody(const std::string& k,
                 std::vector<std::unique_ptr<ASTNode>> m,
                 const std::string& def,
                 const std::optional<std::string>& n = std::nullopt,
                 const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          members(std::move(m)),
          definition(def) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["definition"] = definition;
      j["members"] = json::array();
      for (const auto& m : members)
        j["members"].push_back(m->to_dict());
      return j;
    }

};

class Instance : public ASTNode {
public:
    std::unique_ptr<ASTNode> body;

    Instance(const std::string& k,
             std::unique_ptr<ASTNode> b,
             const std::optional<std::string>& n = std::nullopt,
             const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), body(std::move(b)) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["body"] = body->to_dict();
      return j;
    }

};


class ContinuousAssign;



class Root : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> members;

    Root(const std::string& k,
         std::vector<std::unique_ptr<ASTNode>> m,
         const std::optional<std::string>& n = std::nullopt,
         const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a),
          members(std::move(m)) {}

    json to_dict() const override {
      json j = ASTNode::to_dict();
      j["members"] = json::array();
      for (const auto& m : members)
        j["members"].push_back(m->to_dict());
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


    std::string kind = data.value("kind", "");
   
    std::optional<std::string> name = (data.contains("name") 
    && !data["name"].is_null()) ? std::make_optional(data["name"].get<std::string>()) : std::nullopt;
    
    std::optional<int> addr = (data.contains("addr") && !data["addr"].is_null()) ? std::make_optional(data["addr"].get<int>()) : std::nullopt;

    // Dispatch on the "kind" field
     if (kind == "Assignment"){
      std::string type=data.at("type").get<std::string>();
      bool isNonBlocking=data.at("isNonBlocking").get<bool>();
      // Recursive calls to convert left/right
      auto left=from_dict(data.at("left"));
      auto right=from_dict(data.at("right"));
      //Construct an Assignment
      return std::make_unique<Assignment>(kind,type,std::move(left),std::move(right),isNonBlocking,name,addr);
    } 
   
     else if (kind == "BinaryOp"){
      std::string type=data.at("type").get<std::string>();
      std::string op=data.at("op").get<std::string>();
      auto left=from_dict(data.at("left"));
      auto right=from_dict(data.at("right"));
      return std::make_unique<BinaryOp>(kind,type,op,std::move(left),std::move(right),name,addr);
   } 
    
   else if (kind == "UnaryOp"){
      std::string type=data.at("type").get<std::string>();
      std::string op=data.at("op").get<std::string>();
      auto operand=from_dict(data.at("operand"));
      return std::make_unique<UnaryOp>(kind,type,op,std::move(operand),name,addr);
    }
 
    else if (kind == "CompilationUnit"){
      return std::make_unique<CompilationUnit>(kind,name,addr);
    }

    else if (kind == "ContinuousAssign"){
      auto assignment=from_dict(data.at("assignment"));
      return std::make_unique<ContinuousAssign>(kind,std::move(assignment),name,addr);
    }

  else if (kind == "Conversion") {
    std::string type = data.at("type").get<std::string>();
    auto operand = from_dict(data.at("operand"));

    if (data.contains("constant")) {
        std::string constant = data.at("constant").get<std::string>();
        return std::make_unique<Conversion>(kind, type, std::move(operand), constant, name, addr);
    } else {
        return std::make_unique<Conversion>(kind, type, std::move(operand), std::nullopt, name, addr);
    }
}
 
   
   else if (kind == "Instance") {
         auto body = from_dict(data.at("body"));
         return std::make_unique<Instance>(kind, std::move(body), name, addr);
    }

    else if (kind == "NamedValue"){
      std::string type=data.at("type").get<std::string>();
      std::string symbol=data.at("symbol").get<std::string>();
      std::optional<std::string> constant=(data.contains("constant") && !data["constant"].is_null()) ? std::make_optional(data["constant"].get<std::string>()) : std::nullopt;
      return std::make_unique<NamedValue>(kind,type,symbol,constant,name,addr);
    }

     else if (kind == "Net"){
      std::string type=data.at("type").get<std::string>();
      auto netType=from_dict(data.at("netType"));
      return std::make_unique<Net>(kind,type,std::move(netType),name,addr);
    }
   
    else if (kind == "NetType"){
      std::string type=data.at("type").get<std::string>();
      return std::make_unique<NetType>(kind,type,name,addr);
    }

     else if (kind == "Port"){
      std::string type=data.at("type").get<std::string>();
      std::string direction=data.at("direction").get<std::string>();
      std::string internalSymbol=data.at("internalSymbol").get<std::string>();
      return std::make_unique<Port>(kind,type,direction,internalSymbol,name,addr);
    }
   
    else if (kind == "PrimitiveInstance"){
      std::string primitiveType=data.at("primitiveType").get<std::string>();
      auto ports=from_dict_list(data.at("ports"));
      return std::make_unique<PrimitiveInstance>(kind,primitiveType,std::move(ports),name,addr);
    }

    else if (kind == "Root"){
      auto members=from_dict_list(data.at("members"));
      return std::make_unique<Root>(kind,std::move(members),name,addr);
    }

    else if (kind == "Variable"){
      std::string type=data.at("type").get<std::string>();
      std::string lifetime=data.at("lifetime").get<std::string>();
      return std::make_unique<Variable>(kind,type,lifetime,name,addr);
    }

    else if (kind == "ProceduralBlock"){
      auto body=from_dict(data.at("body"));
      std::string procedureKind=data.at("procedureKind").get<std::string>();
      return std::make_unique<ProceduralBlock>(kind,std::move(body),procedureKind, name, addr);
    }

  else if (kind == "Block") {
    std::string blockKind = data.at("blockKind").get<std::string>();
    auto body = from_dict(data.at("body"));
    std::vector<std::unique_ptr<ASTNode>> bodyVec;
    bodyVec.push_back(std::move(body));  // Use std::move to transfer ownership
    return std::make_unique<Block>(kind, blockKind, std::move(bodyVec), name, addr);
} 
  else if (kind == "ExpressionStatement") {
    auto expr = from_dict(data.at("expr"));
    std::vector<std::unique_ptr<ASTNode>> exprs;
    exprs.push_back(std::move(expr));  // Use std::move to transfer ownership
    return std::make_unique<ExpressionStatement>(kind, std::move(exprs), name, addr);
} 
    else if (kind == "List"){
      auto listNodes=from_dict_list(data.at("list"));
      return std::make_unique<ListNode>(kind,std::move(listNodes),name,addr);
    }

  else if (kind == "IntegerLiteral") {
    std::string type = data.value("type", "");
    std::string value = data.value("value", "");
    std::string constant = data.value("value", "");  // can be same for simplicity
    return std::make_unique<IntegerLiteral>(kind, type, value, constant, name, addr);
  }
 
    else if (kind == "TranslationUnitDecl") {
      auto decls = from_dict_list(data.value("inner", json::array()));
      return std::make_unique<TranslationUnitDecl>(kind, std::move(decls), name, addr);
    }

    else if (kind == "FunctionDecl") {
     std::string retType = data.value("type", "");
     auto inner = from_dict_list(data.value("inner", json::array()));
      return std::make_unique<FunctionDecl>(kind, retType, std::move(inner), name, addr);
    }
    
    else if (kind == "ParmVarDecl") {
      std::string type = data.value("type", "");
      return std::make_unique<ParmVarDecl>(kind, type, name, addr);
    }

    else if (kind == "VarDecl") {
      std::string type = data.value("type", "");
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
      std::string type = data.value("type", "");
      std::string op = data.value("opcode", "unknown");
      auto left = from_dict(data["inner"][0]);
      auto right = from_dict(data["inner"][1]);
      return std::make_unique<BinaryOp>("BinaryOp", type, op, std::move(left), std::move(right), name, addr);
    }
    
    else if (kind == "CallExpr") {
        auto args = from_dict_list(data.value("inner", json::array()));
        return std::make_unique<CallExpr>(kind, std::move(args), name, addr);
    }

    else if (kind == "DeclRefExpr") {
        std::string ref = data.value("referencedDecl", "unknown");
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
        std::string type = data.value("type", "");
        auto inner = from_dict(data["inner"][0]);
        return std::make_unique<ImplicitCastExpr>(kind, type, std::move(inner), name, addr);
    }

    else if (kind == "ParenExpr") {
        auto inner = from_dict(data["inner"][0]);
        return std::make_unique<ParenExpr>(kind, std::move(inner), name, addr);
    }

    else if (kind == "StringLiteral") {
        std::string value = data.value("value", "");
        return std::make_unique<StringLiteral>(kind, value, name, addr);
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



