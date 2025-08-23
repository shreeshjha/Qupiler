#include <string>
#include <regex> 

// Remove operations that are guaranteed to be no-ops:
//  - CX with the same control and target
//  - CCX where one of the controls equals the other control or the target


int removeIdentityOps(std::string &content) {
    int count = 0;
    // 1) q.cx %a[i], %a[i] => no-op 
    {
        std::regex pat(R"((\s*)q\.cx\s+(%\w+\[\d+\])\s*,\s*\2\s*\n)"); 
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
        std::regex pat1(R"((\s*)q\.ccx\s+(%\w+\[\d+\])\s*,\s*\2\s*,\s*(%\w+\[\d+\])\s*\n)");
        std::regex pat2(R"((\s*)q\.ccx\s+(%\w+\[\d+\])\s*,\s*(%\w+\[\d+\])\s*,\s*\2\s*\n)");

        int c1 = 0;
        std::string tmp = content;
        std::smatch m;

        while (std::regex_search(tmp, m, pat1)) {
            ++c1;
            tmp = m.suffix();
        }

        if(c1) {
            count += c1;
            content = std::regex_replace(content, pat1, "$1// OPTIMIZED: removed identity CCX (duplicate controls)\n");
        }

        int c2 = 0;
        tmp = content;
        while (std::regex_search(tmp, m, pat2)) {
            ++c2;
            tmp = m.suffix();
        }

        if(c2) {
            count += c2;
            content = std::regex_replace(content, pat2, 
                "$1// OPTIMIZED: removed identity CCX (control equals target)\n");
        }
    }

    return count;
}
