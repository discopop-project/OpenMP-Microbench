#include <omp.h>
#include "sync_bench.h"
#include "commons.h"
#include <cmath>

std::string bench_name = "SYNC";

void RunBenchmarks() {
    Benchmark(bench_name, "CRITICAL_SECTION", TestCriticalSection, Reference);
    Benchmark(bench_name, "LOCK", TestLock, Reference);
    Benchmark(bench_name, "ATOMIC", TestAtomic, ReferenceAtomic);
    Benchmark(bench_name, "SINGLE", TestSingle, Reference);
    Benchmark(bench_name, "SINGLE_NOWAIT", TestSingleNowait, Reference);
    Benchmark(bench_name, "MASTER", TestMaster, Reference);
    Benchmark(bench_name, "BARRIER", TestBarrier, Reference);
    Benchmark(bench_name, "ORDERED", TestOrdered, Reference);
}


void TestCriticalSection(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel for shared(iterations, workload) default(none) num_threads(threads)
        for (int i = 0; i < iterations; i++) {
            #pragma omp critical (crit)
            {
                DelayFunction(i, workload);
            }
        }
    }
}

void TestLock(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    omp_lock_t lock;
    omp_init_lock(&lock);

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel for shared(iterations, workload, lock) default(none) num_threads(threads)
        for (int i = 0; i < iterations; i++) {
            omp_set_lock(&lock);
            DelayFunction(i, workload);
            omp_unset_lock(&lock);
        }
    }
}

void TestAtomic(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        int j = 0;
        #pragma omp parallel for shared(iterations, workload, j) default(none) num_threads(threads)
        for (int i = 0; i < iterations; i++) {
            DelayFunction(i, workload);
            #pragma omp atomic
            j = j + i;
        }
        if (j < 0) {
            printf("Overflow\n");
        }
    }
}

void TestSingle(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel shared(iterations, workload, threads) default(none) num_threads(threads)
        {
            for (int i = 0; i < ceil((double) iterations / (double) threads); i++) {
                #pragma omp single private(i)
                {
                    DelayFunction(i, workload);
                }
            }
        }
    }
}

void TestSingleNowait(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel shared(iterations, workload, threads) default(none) num_threads(threads)
        {
            for (int i = 0; i < ceil((double) iterations / (double) threads); i++) {
                #pragma omp single nowait private(i)
                {
                    DelayFunction(i, workload);
                }
            }
        }
    }
}

void TestMaster(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel shared(iterations, workload, threads) default(none) num_threads(threads)
        {
            for (int i = 0; i < ceil((double) iterations / (double) threads); i++) {
                #pragma omp master
                {
                    DelayFunction(i, workload);
                }
            }
        }
    }
}

void TestBarrier(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel shared(iterations, workload, threads) default(none) num_threads(threads)
        {
            for (int i = 0; i < ceil((double) iterations / (double) threads); i++) {
                DelayFunction(i, workload);
                #pragma omp barrier
            }
        }
    }
}

void TestOrdered(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel for ordered shared(iterations, workload) default(none) num_threads(threads)
        for (int i = 0; i < iterations; i++) {
            #pragma omp ordered
            DelayFunction(i, workload);
        }
    }
}

void Reference(const DataPoint& data) {
    for (int rep = 0; rep < data.directive; rep++) {
        for (int i = 0; i < data.iterations; i++) {
            DelayFunction(i, data.workload);
        }
    }
}

void ReferenceAtomic(const DataPoint& data) {
    for (int rep = 0; rep < data.directive; rep++) {
        int j = 0;
        for (int i = 0; i < data.iterations; i++) {
            DelayFunction(i, data.workload);
            j = j + i;
        }
        if (j < 0) {
            printf("Overflow\n");
        }
    }
}

int main(int argc, char **argv) {
    ParseArgs(argc, argv);

    if (SAVE_FOR_EXTRAP) {
        RemoveBench(bench_name);
    }

    RunBenchmarks();

    return 0;
}