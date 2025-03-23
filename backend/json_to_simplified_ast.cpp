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
}


int main() {

    return 0;
}



