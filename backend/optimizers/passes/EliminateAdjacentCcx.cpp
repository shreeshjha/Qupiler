#include <string>
#include <regex> 

int eliminateAdjacentCcx(std::string &content) {
    std::regex pattern(R"((\s*)(q\.ccx\s+(%\w+\[\d+\])\s*,\s*(%\w+\[\d+\])\s*,\s*(%\w+\[\d+\]))\s*\n\1\2)");
    
    int count = 0;
    std::smatch m;
    std::string tmp = content;
    while (std::regex_search(tmp, m, pattern)) {
       ++count;
       tmp = m.suffix();
    }
    if(count) {
        content = std::regex_replace(content, pattern,
            "$1// OPTIMIZED: Cancelled adjacent Toffolis ($3,$4->$5)$6");
    }
    return count;
}
