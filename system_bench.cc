#include "system_bench.h"
#include "commons.h"
#include <omp.h>

std::string bench_name = "SYSTEM";

int main(int argc, char **argv){

    PrintCompilerVersion();
    printf("\n");

    if (SAVE_FOR_EXTRAP) {
        RemoveBench(bench_name);
    }

    get_system_information();
    test_system();

    return 0;
}

void test_system(){
    // test GPUs
    for(int i = 0; i < omp_get_num_devices(); i++){
        test_gpu_execution(i);
    }
}

void test_gpu_execution(int device_id){
    printf("Test execution on device %d...\n", device_id);
    int code = -1;
    #pragma omp target device(device_id) map(tofrom: code)
    {
        if (omp_is_initial_device()) {
            // code stays -1 to report a problem in the offloaded execution
        }
        else{
            code = omp_get_device_num();
        }
    }
    if(code == -1){
        printf("\tWARNING! This doesnt seem to be running on a target! \n");
    }
    else{
        printf("\tSuccessful.\n");
    }
}

void get_system_information(){
    printf("# SYSTEM:\n");
    printf("# - devices: %d\n", omp_get_num_devices());
    printf("# - host_device: %d\n", omp_get_initial_device());
    printf("\n");

    get_host_information(omp_get_initial_device());

    for(int device_id = 0; device_id < omp_get_num_devices(); device_id++){
        get_device_information(device_id);
    }
}

void get_host_information(int device_id){
    int num_processors = -1;
    int max_threads = -1;
    // gather stats
    #pragma omp parallel
    #pragma omp single
    {
        num_processors = omp_get_num_procs();
        max_threads = omp_get_max_threads();
    }
    // print stats
    printf("# HOST: %d\n", device_id);
    printf("# - processors: %d\n", num_processors);
    printf("# - threads: %d\n", max_threads);

    printf("\n");
}

void get_device_information(int device_id){
    int num_processors = -1;
    int max_threads = -1;
    int max_teams = -1;
    // gather stats
    #pragma omp target device(device_id) map(tofrom: num_processors, max_teams)
    {
        num_processors = omp_get_num_procs();
        
    }
    // get number ov available teams
    #pragma omp target teams device(device_id) map(tofrom: max_teams)
    {
        max_teams = omp_get_num_teams();
    }

    // print stats
    printf("# DEVICE: %d\n", device_id);
    printf("# - processors: %d\n", num_processors);
    printf("# - teams: %d\n", max_teams);
    printf("\n");
}