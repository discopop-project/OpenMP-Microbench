#include <algorithm>
#include "commons.h"
#include "reduction_bench.h"

// Runs all Benchmarks
void RunBenchmarks();

std::string bench_name = "REDUCTION";
unsigned long long num_tasks = 1;
unsigned long long grainsize = 1;

int main(int argc, char **argv) {

    ParseArgs(argc, argv);

    if (SAVE_FOR_EXTRAP) {
        RemoveBench(bench_name);
    }

    RunBenchmarks();

    return 0;
}

void RunBenchmarks() {
    unsigned long long minimum_iterations = *std::min_element(NUMBER_OF_ITERATIONS.begin(), NUMBER_OF_ITERATIONS.end());

    for (unsigned long long i = 1; i <= minimum_iterations; i = i * 2) {
        num_tasks = i;
        grainsize = i;
        Benchmark(bench_name, "TASKLOOP_NUM_TASKS_" + std::to_string(num_tasks), ReductionTaskloopNumTasks,
                  Reference);
        Benchmark(bench_name, "TASKLOOP_GRAINSIZE_" + std::to_string(grainsize), ReductionTaskloopGrainsize,
                  Reference);
    }

    Benchmark(bench_name, "FOR", ReductionFor, Reference);
    Benchmark(bench_name, "TASKLOOP", ReductionTaskloop, Reference);
    Benchmark(bench_name, "TASK", ReductionTask, Reference);
    Benchmark(bench_name, "TASKGROUP", ReductionTaskgroup, Reference);

}


void ReductionFor(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        float var = 0;

        #pragma omp parallel for reduction(+ : var) shared(iterations, workload) default(none) num_threads(threads)
        for (int i = 0; i < iterations; i++) {
            var = var + DelayFunction(i, workload);
        }
    }
}

void ReductionTask(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        float var = 0;
        #pragma omp parallel for reduction(task, + : var) shared(iterations, workload) default(none) num_threads(threads)
        for (int i = 0; i < iterations; i++) {
            #pragma omp task in_reduction(+ : var) default(none) shared(i, workload)
            {
                var = var + DelayFunction(i, workload);
            }
        }
    }
}

void ReductionTaskgroup(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;
    unsigned long directive = data.directive;

    #pragma omp parallel shared(iterations, workload, directive) default(none) num_threads(threads)
    {
        #pragma omp master
        {
            for (int rep = 0; rep < directive; rep++) {
                float var = 0.0;
                #pragma omp taskgroup task_reduction(+ : var)
                for (int i = 0; i < iterations; i++) {
                    #pragma omp task in_reduction(+ : var) default(none) shared(i, workload)
                    {
                        var = var + DelayFunction(i, workload);
                    }
                }
            }
        }
    }

}

void ReductionTaskloop(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;
    unsigned long directive = data.directive;

    #pragma omp parallel default(none) shared(directive, iterations, workload) num_threads(threads)
    {
        #pragma omp master
        {
            for (int rep = 0; rep < directive; rep++) {
                float var = 0;

                #pragma omp taskloop reduction(+ : var) shared(iterations, workload) default(none)
                for (int i = 0; i < iterations; i++) {
                    var = var + DelayFunction(i, workload);
                }
            }
        }
    }
}

void ReductionTaskloopNumTasks(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;
    unsigned long directive = data.directive;

    #pragma omp parallel default(none) shared(directive, iterations, workload, num_tasks) num_threads(threads)
    {
        #pragma omp master
        {
            for (int rep = 0; rep < directive; rep++) {
                float var = 0;
                #pragma omp taskloop reduction(+ : var) shared(iterations, workload) default(none) num_tasks(num_tasks)
                for (int i = 0; i < iterations; i++) {
                    var = var + DelayFunction(i, workload);
                }
            }
        }
    }
}

void ReductionTaskloopGrainsize(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;
    unsigned long directive = data.directive;

    #pragma omp parallel default(none) shared(directive, iterations, workload, grainsize) num_threads(threads)
    {
        #pragma omp master
        {
            for (int rep = 0; rep < directive; rep++) {
                float var = 0;
                #pragma omp taskloop reduction(+ : var) shared(iterations, workload) default(none) grainsize(grainsize)
                for (int i = 0; i < iterations; i++) {
                    var = var + DelayFunction(i, workload);
                }
            }
        }
    }
}

void Reference(const DataPoint& data) {
    for (int rep = 0; rep < data.directive; rep++) {
        float var = 0;
        for (int i = 0; i < data.iterations; i++) {
            var = var + DelayFunction(i, data.workload);
        }
    }
}

