#include <cmath>
#include "commons.h"
#include "tasktree_bench.h"

void TreeGenLeaves(unsigned int num_children, unsigned long long tasks_to_create, unsigned long long workload);

std::string bench_name = "TASKTREE";

int main(int argc, char **argv) {

    ParseArgsTaskTree(argc, argv);

    // print version of used compiler - makes it easier to spot build errors
    if(!QUIET) {
        #ifdef __clang__
        printf("using %s %d.%d.%d\n", "clang", __clang_major__, __clang_minor__, __clang_patchlevel__);
        #else
        printf("using %s %d.%d.%d\n", "gcc", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
        #endif
    }

    if (SAVE_FOR_EXTRAP) {
        RemoveBench(bench_name);
    }

    BenchmarkTaskTree(bench_name, "TASKTREELEAVES", StartTreeGenLeaves, Reference);

    return 0;
}

void StartTreeGenLeaves(const DataPoint& data) {

    unsigned int num_children = data.child_nodes;
    unsigned long long int tasks_to_create = data.tasks;
    unsigned long long int workload = data.workload;

    #pragma omp parallel firstprivate(num_children, tasks_to_create, workload) default(none)
    {
        #pragma omp master
        {
            TreeGenLeaves(num_children, tasks_to_create, workload);
        }
    }
}

void TreeGenLeaves(unsigned int num_children, unsigned long long int tasks_to_create, unsigned long long int workload) {

    if (tasks_to_create == 1) {
        DelayFunction(0, workload);
    }
    else {
        if (tasks_to_create <= num_children) {
            for (int i = 0; i < tasks_to_create; i++) {
                #pragma omp task firstprivate(num_children, workload) default(none)
                {
                    TreeGenLeaves(num_children, 1, workload);
                }
            }
        }
        else {
            unsigned long long int num_tasks = floor(tasks_to_create / num_children);
            unsigned int remainder = tasks_to_create % num_children;
            for (int i = 0; i < num_children - 1; i++) {
                #pragma omp task firstprivate(num_children, num_tasks, workload) default(none)
                {
                    TreeGenLeaves(num_children, num_tasks, workload);
                }
            }
            #pragma omp task firstprivate(num_children, num_tasks, workload, remainder) default(none)
            {
                TreeGenLeaves(num_children, num_tasks + remainder, workload);
            }
        }
    }
}

void Reference(const DataPoint& data) {
    unsigned long long int tasks = data.tasks;
    unsigned long workload = data.workload;

    #pragma omp parallel shared(tasks, workload) default(none)
    {
        #pragma omp master
        {
            #pragma omp taskloop shared(tasks, workload) default(none)
            for (int i = 0; i < tasks; i++) {
               DelayFunction(i, workload);
            }
        }
    }
}
