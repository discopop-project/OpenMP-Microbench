#ifndef PPT_P4_TASK_H
#define PPT_P4_TASK_H

#include "commons.h"

/// @brief Tasks created by a task loop inside the master thread
/// @param data the configuration for the microbenchmark
void TestTaskLoopMaster(const DataPoint& data);

/// @brief Tasks created by all threads at once
/// @param data the configuration for the microbenchmark
void TestTaskThreads(const DataPoint& data);

/// @brief Tasks created by all threads, barrier after each task creation
/// @param data the configuration for the microbenchmark
void TestTaskBarrier(const DataPoint& data);

/// @brief Tasks created by just the master thread
/// @param data the configuration for the microbenchmark
void TestTaskMaster(const DataPoint& data);

/// @brief Tasks created by all tasks using a for loop
/// @param data the configuration for the microbenchmark
void TestTaskForLoop(const DataPoint& data);

/// @brief Tasks created by a task loop inside the master thread
/// @param data the configuration for the microbenchmark
void TestTaskWait(const DataPoint& data);

/// @brief Tasks are getting created, after a conditional function call
/// @param data the configuration for the microbenchmark
void TestTaskConditionalTrue(const DataPoint& data);

/// @brief Tasks are not getting created, after a conditional function call
/// @param data the configuration for the microbenchmark
void TestTaskConditionalFalse(const DataPoint& data);

/// @brief Reference implementation for task creation overhead
/// @param data the configuration for the reference
void Reference(const DataPoint& data);

#endif //PPT_P4_TASK_H