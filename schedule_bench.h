#ifndef PPT_P4_SCHEDULEBENCH_H
#define PPT_P4_SCHEDULEBENCH_H

#include "commons.h"

/// @brief Scheduling using Static : nonmonotonic with a specified chunk_size
/// @param data the configuration for the microbenchmark
void TestSchedStaticNonmon(const DataPoint& data);

/// @brief Scheduling using Static : monotonic with a specified chunk_size
/// @param data the configuration for the microbenchmark
void TestSchedStaticMon(const DataPoint& data);

/// @brief Scheduling using Dynamic : nonmonotonic with a specified chunk_size
/// @param data the configuration for the microbenchmark
void TestSchedDynamicNonmon(const DataPoint& data);

/// @brief Scheduling using Dynamic : monotonic with a specified chunk_size
/// @param data the configuration for the microbenchmark
void TestSchedDynamicMon(const DataPoint& data);

/// @brief Scheduling using Guided : nonmonotonic with a specified chunk_size
/// @param data the configuration for the microbenchmark
void TestSchedGuidedNonmon(const DataPoint& data);

/// @brief Scheduling using Guided : monotonic with a specified chunk_size
/// @param data the configuration for the microbenchmark
void TestSchedGuidedMon(const DataPoint& data);

/// @brief Scheduling using Auto scheduling
/// @param data the configuration for the microbenchmark
void TestSchedAuto(const DataPoint& data);

/// @brief Different Scheduling schemes, all runtime defined ones, (non)monotonic and chunksize can, but don't have to get defined first
/// @param data the configuration for the microbenchmark
void TestSchedRuntime(const DataPoint& data);

/// @brief Reference serial implementation
/// @param data the configuration for the reference
void Reference(const DataPoint& data);

#endif //PPT_P4_SCHEDULEBENCH_H
