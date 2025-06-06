#include <string>
#include <regex>  // Added this include for regex support

// Detect and fuse a simple ripple-carry adder pattern
int fuseHighLevel(std::string &content) {
    // Pattern for a simple 1-bit adder:
    //   q.cx %a[0], %r[0]
    //   q.cx %b[0], %r[0]
    //   q.ccx %a[0], %b[0], %r[1]
    // →  q.fused_adder %r, %a, %b
    std::regex pat(
        R"((\s*)q\.cx\s+(%\w+)\[(\d+)\]\s*,\s*(%\w+)\[(\d+)\]\s*\n)"
        R"(\1q\.cx\s+(%\w+)\[(\d+)\]\s*,\s*\4\[\5\]\s*\n)"
        R"(\1q\.ccx\s+\2\[\3\]\s*,\s*\6\[\7\]\s*,\s*\4\[(\d+)\])"
    );
    
    int count = 0;
    std::smatch m;
    std::string tmp = content;
    while (std::regex_search(tmp, m, pat)) {
        ++count;
        tmp = m.suffix();
    }
    
    if (count) {
        content = std::regex_replace(
            content, pat,
            "$1// OPTIMIZED: fused 1-bit adder\n"
            "$1q.fused_adder $4, $2, $6\n"
        );
    }
    
    return count;
}
