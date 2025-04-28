#include <string>
#include <vector>
#include <sstream>

// very simple: collect all CX ops into a list, then remove matching commuting pairs
int commuteCancelCx(std::string &content) {
    // for a production pass you'd parse MLIR; here, a toy line-based
    std::vector<std::pair<int,std::string>> ops;
    std::istringstream in(content);
    std::vector<std::string> lines;
    std::string line;
    int idx=0;
    while (std::getline(in, line)) {
        lines.push_back(line);
        if (line.find("q.cx") != std::string::npos) {
            ops.emplace_back(idx, line);
        }
        ++idx;
    }
    std::vector<bool> drop(lines.size(), false);
    int removed=0;
    // find pairs i<j, same operands, and no intervening op touching those operands
    for (int i=0;i<ops.size();++i) {
        auto [li, ai] = ops[i];
        for (int j=i+1;j<ops.size();++j) {
            auto [lj, aj] = ops[j];
            if (ai==aj) {
                // check no intervening line touches %a or %b
                bool conflict=false;
                auto toks = ai.substr(ai.find("q.cx")+4);
                for (int k=li+1;k<lj;++k) {
                    if (lines[k].find(toks.substr(0, toks.find(',')))!=std::string::npos ||
                        lines[k].find(toks.substr(toks.find(',')+1))!=std::string::npos) {
                        conflict=true; break;
                    }
                }
                if (!conflict) {
                    drop[li]=drop[lj]=true;
                    removed+=2;
                }
                break;
            }
        }
    }
    if (removed) {
        std::ostringstream out;
        for (int i=0;i<lines.size();++i)
            if (!drop[i])
                out << lines[i] << "\n";
        content = out.str();
    }
    return removed/2;
}

