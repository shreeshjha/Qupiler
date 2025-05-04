#include <string>
#include <regex>
#include <vector>
#include <sstream>

// Hoist all single-qubit allocs into one big !qreg< N >.
// E.g., replace
//   %a = q.alloc : !qreg<1>
//   %b = q.alloc : !qreg<1>
// with
//   %anc = q.alloc : !qreg<2>
// and rewrite uses %a[i]→%anc[0], %b[i]→%anc[1].
int hoistSingleAlloc(std::string &content) {
    std::smatch m;
    std::regex alloc1(R"(%(\w+)\s*=\s*q\.alloc\s*:\s*!qreg<1>)");
    std::vector<std::string> names;
    std::string tmp = content;
    while (std::regex_search(tmp, m, alloc1)) {
        names.push_back(m[1]);
        tmp = m.suffix();
    }
    
    if (names.size() < 2) return 0;
    
    // build new alloc
    std::string ancName = "%anc";
    int n = (int)names.size();
    std::ostringstream hoist;
    hoist << "  " << ancName << " = q.alloc : !qreg<" << n << ">\n";
    
    // remove old allocs
    content = std::regex_replace(content, alloc1, "");
    
    // insert hoist after the function declaration
    std::regex funcDecl(R"(func\s+@\w+\(\)\s*->\s*\(\)\s*\{\s*\n)");
    if (std::regex_search(content, m, funcDecl)) {
        // Fix: Convert m[0] to string before concatenation
        std::string match_str = m[0].str();
        content = std::regex_replace(content, funcDecl, 
                                   match_str + hoist.str());
    } else {
        // If function declaration not found, insert at top (fallback)
        content = hoist.str() + content;
    }
    
    // rewrite uses
    for (int i = 0; i < names.size(); i++) {
        std::string pattern = names[i] + R"(\[(\d+)\])";
        std::string replacement = ancName + "[" + std::to_string(i) + "]";
        content = std::regex_replace(content, std::regex(pattern), replacement);
    }
    
    return (int)names.size();
}
