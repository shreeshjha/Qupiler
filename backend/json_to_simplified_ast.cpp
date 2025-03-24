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
};

class NamedValue{
  public:
    std::string kind;
    std::string type;
    std::string symbol;
    std::optional<std::string> constant=std::nullopt;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;

    NamedValue(const std::string &k, const std::string &t, const std::string &sym, const std::optional<std::string> &c=std::nullopt,const std::optional<std::string> &n=std::nullopt, const std::optional<int> &a=std::nullopt)
  : kind(k), type(t), symbol(sym), constant(c), name(n), addr(a)
    {}
};

class BinaryOp;
class Conversion;

class Assignment {
public: 
    std::string kind;
    std::string type;   
    NamedValue left;
    std::variant<NamedValue, BinaryOp, Conversion> right;
    bool isNonBlocking;
    std::optional<std::string> name = std::nullopt;
    std::optional<int> addr = std::nullopt;
    
    Assignment(const std::string& k, const std::string& t, const NamedValue& l, const std::variant<NamedValue, BinaryOp, Conversion>& r, bool nonBlocking, const std::optional<std::string> &n = std::nullopt, const std::optional<int>& a = std::nullopt) : kind(k), type(t), left(l), right(r), isNonBlocking(nonBlocking), name(n), addr(a) {}
};

class BinaryOp{
public:
    std::string kind;
    std::string type;
    std::string op;
    std::variant<NamedValue,std::unique_ptr<BinaryOp>> left;
    std::variant<NamedValue,std::unique_ptr<BinaryOp>> right;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;
    
    BinaryOp(const std::string& k,const std::string &t,const std::string& o,std::variant<NamedValue,std::unqiue_ptr<BinaryOp>> l,std::variant<NamedValue,std::unique_ptr<BinaryOp>> r, const std::optional<std::string> &n=std::nullopt,const std::optional<int> &a=std::nullopt) : kind(k), type(t), op(o), left(std::move(l)), right(std::move(r)), name(n), addr(a)

{}
};

class UnaryOp{
  public:
    std::string kind;
    std::string type;
    std::string op;
    
    std::variant<NamedValue, std::unique_ptr<BinaryOp>,Conversion> operand;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;

    UnaryOp(const std::string &k, const std::string &t,const std::string& o, std::variant<NamedValue, std::unique_ptr<BinaryOp>, Conversion> oper, const std::optional<std::string> &n=std::nullopt, const std::optional<int> &a=std::nullopt)
  : kind(k), type(t), op(o), operand(std::move(oper)), name(n), addr(a)
{}
};


class CompilationUnit{
  public:
    std::string kind;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;

    CompilationUnit(const std::string& k,const std::optional<std::string> &n=std::nullopt, const std::optional<int> &a=std::nullopt)
  : kind(k), name(n), addr(a)
    {}
};

class ContinuousAssign{
  public:
    std::string kind;
    Assignment assignment;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;

    ContinuousAssign(const std::string &k,const Assignment& assign, const std::optional<std::string> &n=std::nullopt, const std::optional<int> &a=std::nullopt)
  : kind(k), assignment(assign), name(n), addr(a)
    {}
};


class IntegerLiteral{
  public:
    std::string kind;
    std::string type;
    std::string value;
    std::string constant;
    std::optiona<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;

    IntegerLiteral(const std::string& k, const std::string& t, const std::string& v,const std::string& c,const std::optional<std::string>& n=std::nullopt, const std::optional<int> &a=std::nullopt)
  : kind(k), type(t), value(v), constant(c), name(n), addr(a)
    {}
};

class Conversion{
  public:
    std::string kind;
    std::string type;
    std::variant<std::unique_ptr<Conversion>, NamedValue, IntegerLiteral> operand;
    std::optional<std::string> constant=std::nullopt;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;

    Conversion(const std::string& k, const std::string& t,std::variant<std::unique_ptr<Conversion>, NamedValue, IntegerLiteral> op, const std::optional<std::string>& c=std::nullopt, const std::optional<std::string> &n=std::nullopt, const std::optional<int> &a=std::nullopt)
  : kind(k), type(t), operand(std::move(op)), constant(c), name(n), addr(a)
{}
}; 


class InstanceBody;

class Instance{
  public:
    std::string kind;
    InstanceBody body;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;

    Instance(const std::string &k, const InstanceBody& b, const std::optional<std::string>& n=std::nullopt, const std::optional<int> &a=std::nullopt)
  : kind(k), body(b), name(n), addr(a)
{}
};


class Port{
  public:
    std::string kind;
    std::string type;
    std::string direction;
    std::string internalSymbol;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;

    Port(const std::string& k,const std::string& t,const std::string& d,const std::string& internalSym, const std::optional<std::string>& n=std::nullopt, const std::optional<int> &a=std::nullopt)
    : kind(k), type(t), direction(t), internalSymbol(internalSym) , name(n), addr(a)
    {}
};
class PrimitiveInstance{
  public:
    std::string kind;
    std::string primitiveType;
    std::vector<std::variant<Assignment,NamedValue>> ports;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;

    PrimitiveInstance(const std::string& k, const std::string& pt,const std::vector<std::variant<Assignment,NamedValue>> &p, const std::optional<std::string> &n=std::nullopt, const std::optional<int> &a=std::nullopt)
    : kind(k), primitiveType(pt), ports(p), name(n)l addr(a)
    {}
};
class Variable;
class ContinuousAssign;
class ProceduralBlock;

class NetType{
  public:
    std::string kind;
    std::string type;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;

    NetType(const std::string& k,const std::string& t, const std::optional<std::string>& n=std:nullopt, const std::optional<int> &a=std::nullopt)
            : kind(k), type(t), name(n), addr(a)
    {}
};
class Net{
  public:
    std::string kind;
    std::string type;
    NetType netType;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;

    Net(const std::strin& k,const std::string& t,const NetType& nt, const std::optiona<std::string> &n = std::nullopt, const std::optional<int> &a=std::nullopt)
    : kind(k), type(t), netType(nt), name(n), addr(a)
    {}
};

class InstanceBody{
  public:
    std::string kind;
    std::vector<std::variant<Port,PrimitiveInstance,Variable,ContinuousAssign,ProceduralBlock,Net>> members;
    std::string definiton;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;

    InstanceBody(const std::string& k, const std::vector<std::variant<Port,PrimitiveInstance,Variable,ContinuousAssign,ProceduralBlock, Net>> &m, const std::string &def, const std::optional<std::string> &n=std::nullopt, const std::optional<int> &a=std::nullopt)
  : kind(k),members(m),definiton(def),name(n),addr(a)
  {}
};

class Root {
public:
    std::string kind;
    std::vector<std:variant<CompilationUnit, Instance>> members;
    std::optional<std::string> name=std::nullopt;
    std::optional<int> addr=std::nullopt;
    
    Root(const std::string& k, const std::vector<std::variant<CompilationUnit, Instance>>& m, const std::optional<std::string>& n = std::nullopt, const std::optional<int>& a = std::nullopt) : kind(k), members(m), name(n), addr(a) 
{}
};

class Variable {
public:

    std::string kind;
    std::string type;
    std::string lifetime;
    std::optional<std::string> name = std::nullopt;
    std::optional<int> addr = std::nullopt;
    
    Variable(const std::string& k, const std::string& t, const std::string &l, const std::optional<std::string>& n = std::nullopt, const std::optional<int>& a = std::nullopt) : kind(k), type(t), lifetime(l), name(n), addr(a) 
{}
};


class ExpressionStatement {
public:
    std::string kind;
    std::vector<std::variant<Assignment, BinaryOp, UnaryOp, NamedValue, Conversion>> expr;
    std::optional<std::string> name = std::nullopt;
    std::optional<int> addr = std::nullopt;

    ExpressionStatement(const std::string &k, const std::vector<std::variant<Assignment, BinaryOp, UnaryOp, NamedValue, Conversion>>& e, const std::optional<std::string>& n = std::nullopt, const std::optional<int> &a = std::nullopt) : kind(k), expr(e), name(n), addr(a)
{}
};

class Block {
public:
    std::string kind;
    std::string blockKind;
    std::vector<std::variant<Expression>> body;
    std::optional<std::string> name = std::nullopt;
    std::optional<int> addr = std::nullopt;

    Block(const std::string &k, const std::string &bk, const std::vector<std::variant<Expression>> &b, const std::optional<std::string> &n = std::nullopt, const optional<int> &a = std::nullopt) : kind(k), blockKind(bk), body(b), name(n), addr(a) 
{}
};

class ProceduralBlock {
public:
    std::string kind;
    Block body;
    std::string procedureKind;
    std::optional<std::string> name = std::nullopt;
    std::optional<int> addr = std::nullopt;

    ProceduralBlock(const std::string& k, const Block& b, const std::string& pk, const std::optional<std::string>& n = std::nullopt, 
                    const std::optional<int>& a = std::nullopt) : kind(k), body(b), procedureKind(pk), name(n), addr(a)
{}
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
      auto right=from_doct(data.at("right"));
      //Construct an Assignment
      return std::make_unique<Assignment>(kind,type,std::move(left),isNonBlocking,name,addr);
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
      auto operand=from_dict(data.at("operand"))l
      return std::make_unique<UnaryOp>(kind,type,op,std::move(operand),name,addr)l
    }

    else if (kind == "CompilationUnit"){
      return std::make_unique<CompilationUnit>(kind,name,addr);
    }

    else if (kind == "ContinuousAssign"){
      auto assignment=from_dict(data.at("assignment"));
      return std::make_unique<ContinuousAssign>(kind,std::move(assignment),name,addr);
    }

    else if (kind == "Conversion"){
      std::string type=data.at("type").get<std::string>();
      auto operand=from_dict(data.at("operand"));
      if (data.contains("constant")){
        std::string constant=data.at("constant").get<std::string>();  
        return std::make_unique<Conversion>(kind,type,std::move(operand));
      } 
      else{
        return std::make_unique<Conversion>(kind,type,std::move(operand),name,addr);    
      }
    }

    else if (kind == "Instance"){
      auto body=from_dict(data.at("body"));
      return std::make_unique<Instance>(kind,std::move(body),name,addr);
    }

    else if (kind == "InstanceBody"){
      auto members=from_dict_list(data.at("members"));
      std::string declaration=data.at("definiton").get<std::string>();
      return std::make_unique<InstanceBody>(kind,members,definiton,name,addr);
    }

    else if (kind == "NamedValue"){
      std::string type=data.at("type").get<std::string>();
      std::string symbol=data.at("symbol").get<std::string>();
      std::optional<std::string> constant=(data.contains("constant") && !data["constant"].is_null()) ? std::make_optional(data["constant"].get<std::string()) : std::nullopt;
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
      return std::make_unique<PrimitiveInstance>(kind,primitiveType,ports,name,addr);
    }

    else if (kind == "Root"){
      auto members=from_dict_list(data.at("members"));
      return std::make_unique<Root>(kind,members,name,addr);
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

    else if (kind == "Block"){
      std::string blockKind=data.at("blockKind").get<std::string>();
      auto body=from_dict(data.at("body"));
      return std::make_unique<Block>(kind,blockKind,std::move(body),name,addr);
    }

    else if (kind == "ExpressionStatement"){
      auto expr=from_dict(data.at("expr"));
      return std::make_unique<ExpressionStatement>(kind,std::move(expr),name,addr);
    }

    else if (kind == "List"){
      auto listNodes=from_dict_list(data.at("list"));
      return std::make_unique<ListNode>(kind,listNodes,name,addr);
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
std::vector<std::unique_ptr<ASTNode>> from_dict_list(const json& arr){
  std::vector<std::unique_ptr>> nodes;
  for (const auto& item:arr){
    nodes.push_back(from_dict(item));
  }
  return nodes;
}



int main() {

    return 0;
}



