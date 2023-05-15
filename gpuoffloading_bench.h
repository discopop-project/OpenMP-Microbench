#ifndef PPT_P4_GPUOffloading_H
#define PPT_P4_GPUOffloading_H

#include "commons.h"

/// @brief Runs all Microbenchmarks and sets them up too.
/// The target has to get warmed up first, that's why we run a dummy calculation before the microbenchmarks
void BenchmarkAll();


/// @brief Moves the Data to the GPU and from it and executes
/// @param data the configuration for the microbenchmark
void CopyDataToGpuAndBackSimultaneousExec(const DataPoint& data);

/// @brief Dummy calculation to be run before any microbenchmark, in order to warm up the target
void FirstGpuAction();

/// @brief Moves the Data to the GPU and from it without executing something
/// @param data the configuration for the microbenchmark
void CopyDataToGpuAndBackSeparateExec(const DataPoint& data);


#endif //PPT_P4_GPUOffloading_H
