#include <algorithm>
#include "doall_bench.h"
#include "commons.h"

// Runs all microbenchmarks
void RunBenchmarks();

// allocate variables right away to reduce measured work
unsigned int threads;
unsigned long long int iterations;
unsigned long workload;
unsigned long directive;
unsigned long long array_size = -1;
float* array;
float* array_thread_private;
#pragma omp threadprivate(array_thread_private)

struct ArrayContainer {
public:
    size_t length;
    float arr[];
};


// NOTE: reallocating the array during preprocessing and then deleting
// it during postprocessing for every single benchmark is time-consuming...
// simple solution: reallocate only, if new array size is different
// drawback: on the last execution we do not call delete[]
// this could be solved by better software design, but i can't be bothered with this today
#define REUSE_ARRAY true

/// @brief Sets up a microbenchmark
/// Sets the vector, because we don't want to benchmark on this.
/// Has to get called before every microbenchmark
void Preprocessing(const DataPoint& data) {
    ArrayContainer arrContainer{};
    threads = data.threads;
    iterations = data.iterations;
    workload = data.workload;
    directive = data.directive;
    if(REUSE_ARRAY) {
        if(array_size != data.array_size) {
            delete[] array;
            delete[] array_thread_private;
            array_size = data.array_size;
            array = new float[array_size]();
            array_thread_private = new float[array_size]();
        }// else: array is already correctly allocated
    } else {
        array_size = data.array_size;
        array = new float[array_size]();
        array_thread_private = new float[array_size]();
    }
}

/// @brief Finishes the microbenchmark
/// Deleting the vector, so no memory leaks exist
/// Has to get called after every microbenchmark
void Postprocessing() {
    if(REUSE_ARRAY) {
        // leave the array untouched
    } else {
        delete[] array;
        delete[] array_thread_private;
    }
}

std::string bench_name = "DOALL";


int main(int argc, char **argv) {

    ParseArgs(argc, argv);

    PrintCompilerVersion();

    if (SAVE_FOR_EXTRAP) {
        RemoveBench(bench_name);
    }

    RunBenchmarks();

    return 0;
}

void RunBenchmarks() {
    Benchmark(bench_name, "DOALL", TestDoAll, ReferenceWithoutArray, Preprocessing, Postprocessing);
    Benchmark(bench_name, "FIRSTPRIVATE", TestDoallFirstprivate, ReferenceWithArray, Preprocessing, Postprocessing);
    Benchmark(bench_name, "SHARED", TestDoAllShared, ReferenceWithArray, Preprocessing, Postprocessing);
    Benchmark(bench_name, "SEPARATED", TestDoallSeparated, ReferenceWithArray, Preprocessing, Postprocessing);
    //Benchmark(bench_name, "PRIVATE", TestDoallPrivate, ReferenceWithArray, Preprocessing, Postprocessing);
    //Benchmark(bench_name, "COPYIN", TestCopyin, ReferenceWithArray, Preprocessing, Postprocessing);
    //Benchmark(bench_name, "COPY_PRIVATE", TestCopyPrivate, ReferenceWithArray, Preprocessing, Postprocessing);
}

// TODO we do not even need to pass DataPoint anymore, its all done in preprocessing
void TestDoallFirstprivate(const DataPoint& data) {
    for (int rep = 0; rep < directive; rep++) {
        #pragma omp parallel for num_threads(threads) default(none) shared(iterations, workload) firstprivate(array)
        for (int i = 0; i < iterations; i++) {
            ARRAY_DELAY(workload, array);
        }
    }
}

void TestDoallPrivate(const DataPoint& data) {
    for (int rep = 0; rep < directive; rep++) {
        #pragma omp parallel for num_threads(threads) default(none) shared(iterations, workload) private(array)
        for (int i = 0; i < iterations; i++) {
            ARRAY_DELAY(workload, array);
        }
    }
}

void TestDoallSeparated(const DataPoint& data) {
    #pragma omp parallel num_threads(threads) default(none) shared(directive, iterations, workload, array)
    for (int rep = 0; rep < directive; rep++) {
        #pragma omp for
        for (int i = 0; i < iterations; i++) {
            ARRAY_DELAY(workload, array);
        }
    }
}

void TestDoAllShared(const DataPoint& data) {
    for (int rep = 0; rep < directive; rep++) {
        #pragma omp parallel for num_threads(threads) default(none) shared(iterations, workload, array)
        for (int i = 0; i < iterations; i++) {
            ARRAY_DELAY(workload, array);
        }
    }
}

void TestDoAll(const DataPoint& data) {
    for (int rep = 0; rep < directive; rep++) {
        #pragma omp parallel for num_threads(threads) default(none) shared(iterations, workload, array)
        for (int i = 0; i < iterations; i++) {
            DELAY(workload);
        }
    }
}

void TestCopyin(const DataPoint& data) {
    for (int rep = 0; rep < directive; rep++) {
        #pragma omp parallel for num_threads(threads) default(none) copyin(array_thread_private) shared(iterations, workload)
        for (int i = 0; i < iterations; i++) {
            ARRAY_DELAY(workload, array_thread_private);
        }
    }
}

void TestCopyPrivate(const DataPoint& data) {
    for (int rep = 0; rep < directive; rep++) {
        #pragma omp parallel num_threads(threads) default(none) shared(iterations, workload) private(array)
        {
            #pragma omp single copyprivate(array)
            {
                for (int i = 0; i < iterations; i++) {
                    ARRAY_DELAY(workload, array);
                }
            }
        }
    }
}

void ReferenceWithArray(const DataPoint& data) {
    for (int rep = 0; rep < directive; rep++) {
        for (int i = 0; i < iterations; i++) {
            DELAY(workload)
        }
    }
}


void ReferenceWithoutArray(const DataPoint& data) {
    for (int rep = 0; rep < directive; rep++) {
        for (int i = 0; i < iterations; i++) {
            DELAY(workload);
        }
    }
}
