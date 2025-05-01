#include <string>
#include <regex> 

// Remove operations that are guaranteed to be no-ops:
//  - CX with the same control and target
//  - CCX where one of the controls equals the other control or the target


int removeIdentityOps(std::string &content) {
    int count = 0;
    // 1) q.cx %a, %a => no-op 
    {
        std::regex pat(R"((\s*)q\.cx\s+(\%\w+)\s*,\s*\2\s*\n)");
        std::smatch m;
        std::string tmp = content;
        while(std::regex_search(tmp, m, pat)) {
            ++count;
            tmp = m.suffix();
        }
        if(count) {
            content = std::regex_replace(content, pat, "$1// OPTIMIZED: removed identity CX ($2)\n");
        }
    }

    // 2) q.ccx %a, %a, %b  or q.ccx %a, %b, %a  => no-op
    {
        std::regex pat(R"((\s*)q\.ccx\s+(\%\w+)\s*,\s*\2\s*,\s*(\%\w+))");
        int c2 = std::count_if(
            std::sregex_iterator(content.begin(), content.end(), pat),
            std::sregex_iterator(), [](auto&){return true;}
        );
        if (c2) {
            count += c2;
            content = std::regex_replace(
              content, pat,
              "$1// OPTIMIZED: removed identity CCX (duplicate controls)\n"
            );
        }
    }

    return count;
}
