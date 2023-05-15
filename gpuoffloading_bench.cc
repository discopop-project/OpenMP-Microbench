#include "gpuoffloading_bench.h"
#include "commons.h"
#include <omp.h>

float *vec;
unsigned int team_size;
unsigned long long array_size;
#pragma omp declare target to (vec, team_size, array_size)

std::string bench_name = "GPUOFFLOADING";

/// @brief Delay function for the offloading tasks
/// Has to be used, because the whole function has to get loaded onto the gpu
float LoopInner(unsigned long long i, float* vector, unsigned long long workload);

/// @brief Reference implementation
/// Used for calculating the overhead
void Reference(const DataPoint& data);

int main(int argc, char **argv){

    ParseArgsGpu(argc, argv);

    if (SAVE_FOR_EXTRAP) {
        RemoveBench(bench_name);
    }

    BenchmarkAll();

    return 0;
}

/// @brief Sets up a microbenchmark
/// Sets the vector, because we don't want to benchmark on this.
/// Has to get called before every microbenchmark
void Preprocessing(const DataPoint& data) {
    team_size = data.teams;
    array_size = data.array_size;
    vec = new float[array_size]();
}

/// @brief Finishes the microbenchmark
/// Deleting the vector, so no memory leaks exist
/// Has to get called after every microbenchmark
/// Dummy calculation, to make sure that the vector doesn't get optimized away
void Postprocessing() {
    if(vec[0] < 0) {
        vec[1]++;
    }
    delete[] vec;
}


void BenchmarkAll(){
    FirstGpuAction();

    BenchmarkGpu(bench_name, "COPY_DATA_SIMULTANEOUS_EXEC", Preprocessing,
                 CopyDataToGpuAndBackSimultaneousExec, Postprocessing, Reference);

    BenchmarkGpu(bench_name, "COPY_DATA_SEPARATE_EXEC", Preprocessing,
                 CopyDataToGpuAndBackSeparateExec, Postprocessing, Reference);
}

void FirstGpuAction(){
     
    #pragma omp target 
    {
        if (omp_is_initial_device()) {
            printf("This doesnt seem to be running on a target! \n");
            exit(-1);
        }
    }
    // The first task on a GPU can take a lot longer, this prevents it from influencing the measurements
    DataPoint data{};
    data.iterations = 100;
    data.workload = 100;
    data.teams = 100;
    data.gpu_threads = 100;
    data.directive = 1;
    data.array_size = 200;

    Preprocessing(data);
    CopyDataToGpuAndBackSimultaneousExec(data);
    Postprocessing();
}


void CopyDataToGpuAndBackSimultaneousExec(const DataPoint& data) {
    unsigned int threads = data.gpu_threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        float a = 0.0f;

        #pragma omp target teams distribute parallel for num_teams(team_size) thread_limit(threads) map(tofrom: vec[0:array_size]) shared(iterations, workload, vec, array_size) private(a) default(none)
        for (int i = 0; i < iterations; i++) {
            a = LoopInner(i, vec, workload);
            vec[i % array_size] = a;
        }
    }
}


void CopyDataToGpuAndBackSeparateExec(const DataPoint& data){
    unsigned int threads = data.gpu_threads;
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        #pragma omp target enter data map(to: vec[0:array_size])
        float a = 0.0f;

        #pragma omp target teams distribute parallel for num_teams(team_size) thread_limit(threads) shared(iterations, workload, vec, array_size) private(a) default(none)
        for (int i = 0; i < iterations; i++) {
            a = LoopInner(i, vec, workload);
            vec[i % array_size] = a;
        }

        #pragma omp target exit data map(from: vec[0:array_size])
    }
}


#pragma omp declare target
float LoopInner(unsigned long long i, float *vector, unsigned long long workload) {
    float a = (float) i;
    for (int j = 0; j < workload; j++) {
        a += (float) j;
    }
    if (a < 0) {
        printf("%f \n", a + (float) (vector[i]));
    }
    return a;
}
#pragma omp end declare target

void Reference(const DataPoint& data) {
    unsigned long long int iterations = data.iterations;
    unsigned long workload = data.workload;

    for (int rep = 0; rep < data.directive; rep++) {
        float a = 0.0f;
        #pragma omp parallel for shared(iterations, workload, vec, array_size) num_threads(data.threads) private(a) default(none)
        for (unsigned long long i = 0; i < iterations; i++) {
            a = LoopInner(i, vec, workload);
            vec[i % array_size] = a;
        }
    }
}