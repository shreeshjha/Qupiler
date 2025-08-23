#include <string>
#include <regex>

// return number of removals
//
int eliminateAdjacentCx(std::string &content) {
    // std::regex pattern(R"((\s*)(q\.cx\s+(%\w+)\s*,\s*(%\w+))(\s*)\n\1(q\.cx\s+\3\s*,\s*\4)(\s*))");
    std::regex pattern(R"((\s*)(q\.cx\s+(%\w+\[\d+\])\s*,\s*(%\w+\[\d+\]))(\s*)\n\1\2)");

    int count = 0;
    std::smatch m;
    std::string tmp = content;
    while (std::regex_search(tmp, m, pattern)) {
        ++count;
        tmp = m.suffix();
    }

    if(count) {
        content = std::regex_replace(content, pattern, 
                    "$1// OPTIMIZED: Cancelled adjacent CNOTs ($3,$4)$5");
    }
    return count;
}
