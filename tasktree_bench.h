#ifndef PPT_P4_TASKTREE_H
#define PPT_P4_TASKTREE_H


/// @brief Tasks are getting created in a Tree style configuration. Every node is a new Task, but only the leaves have an actual
/// Workload. The tree being created is an M-ary balanced tree with N leaves in total. M being the number of child nodes
/// and N being the Number of Tasks to be created
/// @param data the configuration for the microbenchmark
void StartTreeGenLeaves(const DataPoint& data);

/// @brief Reference implementation, N tasks are being created by a task loop.
/// @param data the configuration for the reference
void Reference(const DataPoint& data);

#endif //PPT_P4_TASKTREE_H
