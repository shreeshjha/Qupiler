// ExtendedConstantFolding.cpp
#include <string>
#include <regex>

// Fold addition or subtraction by zero:
//   q.init %r, 0
//   q.add  %d, %r, %r   =>  q.init %d, 0
int foldAddZero(std::string &content) {
    std::regex pat(R"((\s*)q\.init\s+(%\w+),\s*0\s*\n\1q\.add\s+(%\w+),\s*\2\s*,\s*\2)");
    int count = 0;
    std::string tmp = content;
    std::smatch m;
    while (std::regex_search(tmp, m, pat)) {
        ++count;
        tmp = m.suffix();
    }
    if (count) {
        content = std::regex_replace(
          content, pat,
          "$1// OPTIMIZED: folded add zero\n$1q.init $3, 0"
        );
    }
    return count;
}

