#include <iostream>
#include <stdio.h>
#include <string>
#include <optional>
#include <variant>
#include <vector>
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
};


class CompilationUnit : public ASTNode {
public:
    CompilationUnit(const std::string& k,
                    const std::optional<std::string>& n = std::nullopt,
                    const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a) {}
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
};

class Instance : public ASTNode {
public:
    std::unique_ptr<ASTNode> body;

    Instance(const std::string& k,
             std::unique_ptr<ASTNode> b,
             const std::optional<std::string>& n = std::nullopt,
             const std::optional<int>& a = std::nullopt)
        : ASTNode(k, n, a), body(std::move(b)) {}
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

    else if (kind == "IntegerLiteral"){
      std::string type=data.at("type").get<std::string>();
      std::string value=data.at("value").get<std::string>();
      std::string constant=data.at("constant").get<std::string>();
      return std::make_unique<IntegerLiteral>(kind,type,value,constant,name,addr);
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



int main() {

    return 0;
}



