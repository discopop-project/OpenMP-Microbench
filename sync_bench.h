#ifndef PPT_P4_SYNCBENCH_H
#define PPT_P4_SYNCBENCH_H

#include "commons.h"

/// @brief Synchronisation with the computation being wrapped in a critical section
/// @param data the configuration for the microbenchmark
void TestCriticalSection(const DataPoint& data);

/// @brief Synchronisation with the computation being wrapped in a locked section
/// @param data the configuration for the microbenchmark
void TestLock(const DataPoint& data);

/// @brief Synchronisation with an atomic arithmetic operation on a shared variable after the computation
/// @param data the configuration for the microbenchmark
void TestAtomic(const DataPoint& data);

/// @brief Synchronisation with just one thread doing the calculations
/// @param data the configuration for the microbenchmark
void TestSingle(const DataPoint& data);

/// @brief Synchronisation with just one thread doing the calculations, nowait clauses added
/// @param data the configuration for the microbenchmark
void TestSingleNowait(const DataPoint& data);

/// @brief Synchronisation with the master thread doing the calculations
/// @param data the configuration for the microbenchmark
void TestMaster(const DataPoint& data);

/// @brief Synchronisation with barrier after every loop iteration
/// @param data the configuration for the microbenchmark
void TestBarrier(const DataPoint& data);

/// @brief Synchronisation with the DoAll loop being ordered and ordered directive
/// @param data the configuration for the microbenchmark
void TestOrdered(const DataPoint& data);

/// @brief Reference implementation for overhead calculation
/// @param data the configuration for the reference
void Reference(const DataPoint& data);

/// @brief Reference implementation for overhead calculation for the atomic directive
/// @param data the configuration for the reference
void ReferenceAtomic(const DataPoint& data);

#endif //PPT_P4_SYNCBENCH_H
