#include "schedule_bench.h"
#include "commons.h"
#include <omp.h>
#include <algorithm>

std::string bench_name = "SCHEDULE";

int chunk_size = 1;

void RunRuntimeBenchmarks() {
    unsigned long long minimum_iterations = *std::min_element(NUMBER_OF_ITERATIONS.begin(), NUMBER_OF_ITERATIONS.end());

    for (unsigned long long chunk = 1; chunk <= minimum_iterations; chunk = chunk * 2) {
        chunk_size = chunk;

        omp_set_schedule(omp_sched_static, 0);
        Benchmark(bench_name, "RUNTIME_STATIC_NON_MON_" + std::to_string(chunk), TestSchedRuntime, Reference);

        omp_set_schedule(static_cast<omp_sched_t>(omp_sched_static + omp_sched_monotonic), 0);
        Benchmark(bench_name, "RUNTIME_STATIC_MON_" + std::to_string(chunk), TestSchedRuntime, Reference);

        omp_set_schedule(omp_sched_dynamic, 0);
        Benchmark(bench_name, "RUNTIME_DYNAMIC_NON_MON_" + std::to_string(chunk), TestSchedRuntime, Reference);

        omp_set_schedule(static_cast<omp_sched_t>(omp_sched_dynamic + omp_sched_monotonic), 0);
        Benchmark(bench_name, "RUNTIME_DYNAMIC_MON_" + std::to_string(chunk), TestSchedRuntime, Reference);

        omp_set_schedule(omp_sched_guided, 0);
        Benchmark(bench_name, "RUNTIME_GUIDED_NON_MON_" + std::to_string(chunk), TestSchedRuntime, Reference);

        omp_set_schedule(static_cast<omp_sched_t>(omp_sched_guided + omp_sched_monotonic), 0);
        Benchmark(bench_name, "RUNTIME_GUIDED_MON_" + std::to_string(chunk), TestSchedRuntime, Reference);
    }
    omp_set_schedule(omp_sched_auto, 0);
    Benchmark(bench_name, "RUNTIME_AUTO", TestSchedRuntime, Reference);

}

void RunBenchmarks() {
    // This is how the calls for the Benchmark could look like
    unsigned long long minimum_iterations = *std::min_element(NUMBER_OF_ITERATIONS.begin(), NUMBER_OF_ITERATIONS.end());

    for (unsigned long long chunk = 1; chunk <= minimum_iterations; chunk = chunk * 2) {
        Benchmark(bench_name, "STATIC_NON_MON_" + std::to_string(chunk), TestSchedStaticNonmon, Reference);
        Benchmark(bench_name, "STATIC_MON_" + std::to_string(chunk), TestSchedStaticMon, Reference);

        Benchmark(bench_name, "DYNAMIC_NON_MON_" + std::to_string(chunk), TestSchedDynamicNonmon, Reference);
        Benchmark(bench_name, "DYNAMIC_MON_" + std::to_string(chunk), TestSchedDynamicMon, Reference);

        Benchmark(bench_name, "GUIDED_NON_MON_" + std::to_string(chunk), TestSchedGuidedNonmon, Reference);
        Benchmark(bench_name, "GUIDED_MON_" + std::to_string(chunk), TestSchedGuidedMon, Reference);

    }
    Benchmark(bench_name, "AUTO", TestSchedAuto, Reference);
}

void TestSchedStaticNonmon(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned int directive = data.directive;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    #pragma omp parallel default(none) shared(iterations, workload, directive, chunk_size) num_threads(threads)
    {
        for (int rep = 0; rep < directive; rep++) {
            #pragma omp for schedule(nonmonotonic : static, chunk_size)
            for (int i = 0; i < iterations; i++) {
                DelayFunction(i, workload);
            }
        }
    }
}


void TestSchedStaticMon(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned int directive = data.directive;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    #pragma omp parallel default(none) shared(iterations, workload, directive, chunk_size) num_threads(threads)
    {
        for (int rep = 0; rep < directive; rep++) {
            #pragma omp for schedule(monotonic : static, chunk_size)
            for (int i = 0; i < iterations; i++) {
                DelayFunction(i, workload);
            }
        }
    }

}

void TestSchedDynamicNonmon(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned int directive = data.directive;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    #pragma omp parallel default(none) shared(iterations, workload, directive, chunk_size) num_threads(threads)
    {
        for (int rep = 0; rep < directive; rep++) {
            #pragma omp for schedule(nonmonotonic : dynamic, chunk_size)
            for (int i = 0; i < iterations; i++) {
                DelayFunction(i, workload);
            }
        }
    }
}

void TestSchedDynamicMon(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned int directive = data.directive;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    #pragma omp parallel default(none) shared(iterations, workload, directive, chunk_size) num_threads(threads)
    {
        for (int rep = 0; rep < directive; rep++) {
            #pragma omp for schedule(monotonic : dynamic, chunk_size)
            for (int i = 0; i < iterations; i++) {
                DelayFunction(i, workload);
            }
        }
    }
}

void TestSchedGuidedNonmon(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned int directive = data.directive;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    #pragma omp parallel default(none) shared(iterations, workload, directive, chunk_size) num_threads(threads)
    {
        for (int rep = 0; rep < directive; rep++) {
            #pragma omp for schedule(nonmonotonic : guided, chunk_size)
            for (int i = 0; i < iterations; i++) {
                DelayFunction(i, workload);
            }
        }
    }
}

void TestSchedGuidedMon(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned int directive = data.directive;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    #pragma omp parallel default(none) shared(iterations, workload, directive, chunk_size) num_threads(threads)
    {
        for (int rep = 0; rep < directive; rep++) {
            #pragma omp for schedule(monotonic : guided, chunk_size)
            for (int i = 0; i < iterations; i++) {
                DelayFunction(i, workload);
            }
        }

    }
}

void TestSchedAuto(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned int directive = data.directive;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    #pragma omp parallel default(none) shared(iterations, workload, directive) num_threads(threads)
    {
        for (int rep = 0; rep < directive; rep++) {
            #pragma omp for schedule(auto)
            for (int i = 0; i < iterations; i++) {
                DelayFunction(i, workload);
            }
        }
    }
}

void TestSchedRuntime(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned int directive = data.directive;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    omp_sched_t schedule;
    int dummy_chunk;
    omp_get_schedule(&schedule, &dummy_chunk);
    omp_set_schedule(schedule, chunk_size);

    #pragma omp parallel default(none) shared(iterations, workload, directive) num_threads(threads)
    {
        for (int rep = 0; rep < directive; rep++) {
            #pragma omp for schedule(runtime)
            for (int i = 0; i < iterations; i++) {
                DelayFunction(i, workload);
            }
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


int main(int argc, char **argv) {
    ParseArgs(argc, argv);

    if (SAVE_FOR_EXTRAP) {
        RemoveBench(bench_name);
    }

    RunBenchmarks();
    RunRuntimeBenchmarks();

    return 0;
}