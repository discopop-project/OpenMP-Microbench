#include <cmath>
#include "task_bench.h"
#include "commons.h"

void RunBenchmarks();

std::string bench_name = "TASK";

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

    Benchmark(bench_name, "TASKLOOP_MASTER", TestTaskLoopMaster, Reference);
    Benchmark(bench_name, "MASTER", TestTaskMaster, Reference);
    Benchmark(bench_name, "BARRIER", TestTaskBarrier, Reference);
    Benchmark(bench_name, "THREADS", TestTaskThreads, Reference);
    Benchmark(bench_name, "FOR_LOOP", TestTaskForLoop, Reference);
    Benchmark(bench_name, "WAIT", TestTaskWait, Reference);
    Benchmark(bench_name, "CONDITIONAL_TRUE", TestTaskConditionalTrue, Reference);
    Benchmark(bench_name, "CONDITIONAL_FALSE", TestTaskConditionalFalse, Reference);

}


//Tasks created by Master
void TestTaskLoopMaster(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;
    unsigned long directive = data.directive;

    #pragma omp parallel num_threads(threads) shared(iterations, workload, threads, directive) default(none)
    {
        #pragma omp master
        {
            for (int rep = 0; rep < directive; rep++) {
                #pragma omp taskloop shared(iterations, workload, threads) default(none)
                for (int i = 0; i < iterations; i++) {
                    DELAY(workload, i);
                }
            }
        }
    }
}

//One Thread creates all the Tasks 
void TestTaskMaster(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel num_threads(threads) shared(iterations, workload, threads) default(none)
        {
            #pragma omp master
            {
                for (int i = 0; i < iterations; i++) {
                    #pragma omp task shared(i, workload) default(none)
                    {
                        DELAY(workload, i);
                    }
                }
            }
        }
    }
}

//Tasks created by all threads with barrier at end
void TestTaskBarrier(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;
    unsigned long directive = data.directive;

    #pragma omp parallel num_threads(threads) shared(iterations, workload, threads, directive) default(none)
    {
        for (int rep = 0; rep < directive; rep++) {
            for (int i = 0; i < ceil(iterations/threads); i++) {
                #pragma omp task shared(i, workload) default(none)
                {
                    DELAY(workload, i);
                }
                #pragma omp barrier
            }
        }
    }
}

void TestTaskThreads(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel num_threads(threads) shared(iterations, workload, threads) default(none)
        {
            for (int i = 0; i < ceil(iterations/threads); i++) {
                #pragma omp task shared(i, workload) default(none)
                {
                    DELAY(workload, i);
                }
            }
        }
    }
}

void TestTaskForLoop(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel for num_threads(threads) shared(iterations, workload, threads) default(none)
        for (int i = 0; i < iterations; i++) {
            #pragma omp task shared(i, workload) default(none)
            {
                DELAY(workload, i);
            }
        }
    }
}

void TestTaskWait(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {

        #pragma omp parallel num_threads(threads) shared(iterations, workload, threads) default(none)
        {
            for (int i = 0; i < ceil(iterations/threads); i++) {
                #pragma omp task shared(i, workload) default(none)
                {
                    DELAY(workload, i);
                }
                #pragma omp taskwait
            }
        }
    }
}


void TestTaskConditionalTrue(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel num_threads(threads) shared(iterations, workload, threads) default(none)
        {
            for (int i = 0; i < ceil(iterations/threads); i++) {
                #pragma omp task shared(i, workload) if(ReturnTrue()) default(none)
                {
                    DELAY(workload, i);
                }
            }
        }
    }
}

void TestTaskConditionalFalse(const DataPoint& data) {
    unsigned int threads = data.threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp parallel num_threads(threads) shared(iterations, workload, threads) default(none)
        {
            for (int i = 0; i < ceil(iterations/threads); i++) {
                #pragma omp task shared(i, workload) if(ReturnFalse()) default(none)
                {
                    DELAY(workload, i);
                }
            }
        }
    }
}

void Reference(const DataPoint& data) {
    unsigned int threads = data.threads; // // not used, only here for equal amount of work in Test and Reference
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        for (int i = 0; i < iterations; i++) {
            DELAY(workload, i);
        }
    }
}
