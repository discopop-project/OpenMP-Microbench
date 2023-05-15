#ifndef PPT_P4_REDUCTION_H
#define PPT_P4_REDUCTION_H

#include "commons.h"

/// @brief Reduction clause with a DoAll loop
/// @param data the configuration for the microbenchmark
void ReductionFor(const DataPoint& data);

/// @brief Reduction with a Taskloop
/// @param data the configuration for the microbenchmark
void ReductionTaskloop(const DataPoint& data);

/// @brief Reduction with a taskloop with a custom number of total tasks
/// @param data the configuration for the microbenchmark
void ReductionTaskloopNumTasks(const DataPoint& data);

/// @brief Reduction with a taskloop with a custom grainsize
/// @param data the configuration for the microbenchmark
void ReductionTaskloopGrainsize(const DataPoint& data);

/// @brief Reduction with multiple tasks
/// @param data the configuration for the microbenchmark
void ReductionTask(const DataPoint& data);

/// @brief Reduction with a tasks in a taskgroup
/// @param data the configuration for the microbenchmark
void ReductionTaskgroup(const DataPoint& data);

/// @brief Reference implementation of a variable aggregation, for overhead calculation
/// @param data the configuration for the reference
void Reference(const DataPoint& data);

#endif //PPT_P4_REDUCTION_H



