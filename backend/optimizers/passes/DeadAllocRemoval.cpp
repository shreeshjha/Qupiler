#include <string>
#include <set>
#include <sstream>
#include <vector>
#include <regex>

// returns number of allocs removed
int removeDeadAllocs(std::string &content) {
    std::istringstream in(content), out;
    std::vector<std::string> lines;
    std::string line;
    std::set<std::string> used;
    std::vector<std::pair<int,std::string>> allocs;

    std::regex allocRe(R"(%(\w+)\s*=\s*q\.alloc)");

    // first pass: collect allocs and uses
    int idx=0;
    while (std::getline(in, line)) {
        lines.push_back(line);
        std::smatch m;
        if (std::regex_search(line, m, allocRe)) {
            allocs.emplace_back(idx, m[1]);
        }
        // find any %foo[...] uses
        std::regex useRe(R"(%(\w+)\[\d+\])");
        auto begin = std::sregex_iterator(line.begin(), line.end(), useRe);
        auto end   = std::sregex_iterator();
        for (auto it = begin; it!=end; ++it)
            used.insert((*it)[1]);
        ++idx;
    }

    // build new content, skipping dead allocs
    int removed=0;
    std::ostringstream oss;
    for (int i=0; i<lines.size(); ++i) {
        bool drop=false;
        for (auto &p: allocs) {
            if (p.first==i && used.count(p.second)==0) {
                ++removed;
                drop=true;
                break;
            }
        }
        if (!drop) oss << lines[i] << '\n';
    }
    if (removed) content = oss.str();
    return removed;
}

