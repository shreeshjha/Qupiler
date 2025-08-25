#include <string>
#include <regex>

int eliminateAdjacentX(std::string &content) {
    // matches e.g. " q.x %q0[2]\n   q.x %q0[2]"
    std::regex pattern(R"((\s*)(q\.x\s+(%\w+\[\d+\]))\s*\n\1\2)");
    int count = 0;
    std::smatch m;
    std::string tmp = content;
    while (std::regex_search(tmp, m, pattern)) {
        ++count;
        tmp = m.suffix();
    }
    if(count) {
        content = std::regex_replace(content, pattern, 
                    "$1// OPTIMIZED: Cancelled adjacent X ($3)");

    }
    return count;
}


