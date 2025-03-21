#This is just an initilization of json to simplified ast tool

#include <iostream>
#include <stdio.h>
#include <string>
#include <optional>
#include <variant>
#include "./json.hpp"

class ASTNode {
public: 
    std::string kind;
    std::optional<std::string> name = std::nullopt;
    std::optional<int> addr = std::nullopt;

    // This is completely optional and may not be required
    ASTNode(const std::string& k, const std::optional<std::string>& n = std::nullopt, const std::optional<int> &a = std::nullopt) : kind(k), name(n), addr(a) {}
};

class NamedValue;
class BinaryOp;
class Converstion;

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

int main() {

    return 0;
}



