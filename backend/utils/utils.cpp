#include "utils.hpp"
#include <unordered_map>

static std::unordered_map<std::string, int> tmp_counters;

std::string new_tmp(const std::string& prefix) {
    int id = tmp_counters[prefix]++;
    return prefix + std::to_string(id);
}

void reset_tmp_counters() {
    tmp_counters.clear();
}

