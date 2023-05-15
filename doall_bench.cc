#include <algorithm>
#include "doall_bench.h"
#include "commons.h"

// In case this doesn't get compiled with the Environment Variables set by CMake and Make
#ifndef ARRAY_SIZE
int ARRAY_SIZE = 1
#endif

// Runs all microbenchmarks
void RunBenchmarks();

float array[ARRAY_SIZE];
float array_thread_private[ARRAY_SIZE];
#pragma omp threadprivate (array_thread_private)

std::string bench_name = "DOALL_" + std::to_string(ARRAY_SIZE);

int main(int argc, char **argv) {

    ParseArgs(argc, argv);

    if (SAVE_FOR_EXTRAP) {
        RemoveBench(bench_name);
    }

    RunBenchmarks();

    return 0;
}

void RunBenchmarks() {

    Benchmark(bench_name, "DOALL", TestDoAll, Reference);
    Benchmark(bench_name, "SHARED", TestDoAllShared, Reference);
    Benchmark(bench_name, "SEPARATED", TestDoallSeparated, Reference);
    Benchmark(bench_name, "FIRSTPRIVATE", TestDoallFirstprivate, Reference);
    Benchmark(bench_name, "PRIVATE", TestDoallPrivate, Reference);
    Benchmark(bench_name, "COPYIN", TestCopyin, Reference);
    Benchmark(bench_name, "COPY_PRIVATE", TestCopyPrivate, Reference);

}

void TestDoallFirstprivate(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel for num_threads(threads) default(none) shared(iterations, workload) firstprivate(array)
        for (int i = 0; i < iterations; i++) {
            ArrayDelayFunction(i, workload, array);
        }
    }
}

void TestDoallPrivate(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel for num_threads(threads) default(none) shared(iterations, workload) private(array)
        for (int i = 0; i < iterations; i++) {
            ArrayDelayFunction(i, workload, array);
        }
    }
}

void TestDoallSeparated(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;
    unsigned long directive = data.directive;

    #pragma omp parallel num_threads(threads) default(none) shared(directive, iterations, workload, array)
    for (int rep = 0; rep < directive; rep++) {
        #pragma omp for
        for (int i = 0; i < iterations; i++) {
            ArrayDelayFunction(i, workload, array);
        }
    }
}

void TestDoAllShared(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel for num_threads(threads) default(none) shared(iterations, workload, array)
        for (int i = 0; i < iterations; i++) {
            ArrayDelayFunction(i, workload, array);
        }
    }
}

void TestDoAll(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel for num_threads(threads) default(none) shared(iterations, workload, array)
        for (int i = 0; i < iterations; i++) {
            DelayFunction(i, workload);
        }
    }
}

void TestCopyin(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel for num_threads(threads) default(none) copyin(array_thread_private) shared(iterations, workload)
        for (int i = 0; i < iterations; i++) {
            ArrayDelayFunction(i, workload, array_thread_private);
        }
    }
}

void TestCopyPrivate(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel num_threads(threads) default(none) shared(iterations, workload) private(array)
        {
            #pragma omp single copyprivate(array)
            {
                for (int i = 0; i < iterations; i++) {
                    ArrayDelayFunction(i, workload, array);
                }
            }
        }
    }
}

void Reference(const DataPoint& data) {
    for (int rep = 0; rep < data.directive; rep++) {
        for (int i = 0; i < data.iterations; i++) {
            ArrayDelayFunction(i, data.workload, array);
        }
    }
}
