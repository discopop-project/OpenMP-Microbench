#ifndef PPT_P4_DOALL_H
#define PPT_P4_DOALL_H

#include "commons.h"

/// @brief Benchmark for an array in combination with shared clause and parallel for separated
/// @param data the configuration for the microbenchmark
void TestDoallSeparated(const DataPoint& data);

/// @brief Benchmark for an array in combination with shared clause, with cache misses
/// @param data the configuration for the microbenchmark
void TestDoAllShared(const DataPoint& data);

/// @brief Benchmark for no array during calculation, array still is included in shared clause
/// @param data the configuration for the microbenchmark
void TestDoAll(const DataPoint& data);

/// @brief Benchmark for an array in combination with the firstprivate clause
/// @param data the configuration for the microbenchmark
void TestDoallFirstprivate(const DataPoint& data);

/// @brief Benchmark for an array in combination with the private clause
/// @param data the configuration for the microbenchmark
void TestDoallPrivate(const DataPoint& data);

/// @brief Benchmark for an array in combination with the copyin clause
/// @param data the configuration for the microbenchmark
void TestCopyin(const DataPoint& data);

/// @brief Benchmark for an array in combination with the copyprivate clause, only works in a single construct
/// @param data the configuration for the microbenchmark
void TestCopyPrivate(const DataPoint& data);

/// @brief Reference Implementation, for calculating the overhead
/// @param data the configuration for the reference
void Reference(const DataPoint& data);

#endif //PPT_P4_DOALL_H
