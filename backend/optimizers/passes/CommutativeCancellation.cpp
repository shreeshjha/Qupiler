#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <regex>

// very simple: collect all CX ops into a list, then remove matching commuting pairs
/*
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
*/

// Enhanced commutative cancellation: collect all CX ops into a list, 
// then remove matching commuting pairs more aggressively
int commuteCancelCx(std::string &content) {
    // Extract line-by-line for a more detailed analysis
    std::istringstream in(content);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    
    // Parse operations
    std::vector<std::tuple<int, std::string, std::string, std::string>> ops;
    std::regex cxRe(R"(^\s*q\.cx\s+%(\w+)\[(\d+)\],\s*%(\w+)\[(\d+)\])");
    
    for (int i = 0; i < lines.size(); ++i) {
        std::smatch m;
        if (std::regex_search(lines[i], m, cxRe)) {
            std::string ctrl = m[1].str() + "[" + m[2].str() + "]";
            std::string targ = m[3].str() + "[" + m[4].str() + "]";
            ops.emplace_back(i, "cx", ctrl, targ);
        }
    }
    
    // Find cancellable pairs
    std::vector<bool> drop(lines.size(), false);
    int removed = 0;
    
    // Enhanced detection for non-consecutive cancellation
    for (int i = 0; i < ops.size(); ++i) {
        auto [li, op_i, ctrl_i, targ_i] = ops[i];
        if (drop[li]) continue;  // Skip if already marked for removal
        
        for (int j = i + 1; j < ops.size(); ++j) {
            auto [lj, op_j, ctrl_j, targ_j] = ops[j];
            if (drop[lj]) continue;  // Skip if already marked for removal
            
            // Check if operations are identical
            if (op_i == op_j && ctrl_i == ctrl_j && targ_i == targ_j) {
                // Check if gates commute with all operations in between
                bool can_commute = true;
                
                for (int k = i + 1; k < j; ++k) {
                    auto [lk, op_k, ctrl_k, targ_k] = ops[k];
                    if (drop[lk]) continue;  // Skip if already marked for removal
                    
                    // CX gates don't commute if they share a target or if one's target is the other's control
                    if ((targ_i == targ_k) || 
                        (targ_i == ctrl_k) || 
                        (ctrl_i == targ_k)) {
                        can_commute = false;
                        break;
                    }
                }
                
                if (can_commute) {
                    // Mark both operations for removal
                    drop[li] = drop[lj] = true;
                    removed += 2;
                    
                    // Add a comment to show the cancellation
                    lines[li] = std::string("    // OPTIMIZED: Cancelled commuting CX gates (") + 
                                ctrl_i + ", " + targ_i + ")";
                    lines[lj] = "";  // Remove second occurrence
                    
                    break;  // Continue with next operation
                }
            }
        }
    }
    
    if (removed > 0) {
        // Rebuild the content
        std::ostringstream out;
        for (int i = 0; i < lines.size(); ++i) {
            if (!drop[i] || lines[i].find("// OPTIMIZED") != std::string::npos) {
                out << lines[i] << "\n";
            }
        }
        content = out.str();
    }
    
    return removed / 2;  // Return pairs removed
}

