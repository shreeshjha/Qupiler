// backend/passes/EliminateAdjacentX.h
#ifndef ELIMINATE_ADJACENT_X_H
#define ELIMINATE_ADJACENT_X_H

#include <string>

// Scans `content` for back-to-back q.x lines and removes one of each pair.
// Returns number of removals.
int eliminateAdjacentX(std::string &content);

#endif // ELIMINATE_ADJACENT_X_H

