#include <string>
#include <regex>

// fold patterns like:
//   q.init %r, 0
//   q.sub  %d, %r, %r
// →  q.init %d, 0
int foldInitSubZero(std::string &content) {
    //std::regex pattern(R"((\s*)q\.init\s+(%\w+),\s*0\s*\n\1q\.sub\s+(%\w+),\s*\2,\s*\2)");
    std::regex pattern(R"((\s*)q\.init\s+(%\w+\[\d+\])\s*,\s*0[^\n]*\n\1q\.sub\s+(%\w+)\s*,\s*\2\s*,\s*\2)");

    int count=0;
    std::smatch m;
    std::string tmp = content;
    while (std::regex_search(tmp, m, pattern)) {
        ++count;
        tmp = m.suffix();
    }
    if (count) {
        content = std::regex_replace(content, pattern,
            "$1// OPTIMIZED: folded zero-sub\n$1q.init $3, 0");
    }
    return count;
}

