#include "system_bench.h"
#include "commons.h"
#include <omp.h>
#include <iostream>
#include <unistd.h>

std::string bench_name = "SYSTEM";

int main(int argc, char **argv){

    PrintCompilerVersion();
    printf("\n");

    if (SAVE_FOR_EXTRAP) {
        RemoveBench(bench_name);
    }

    test_system();

    printf("\n#################\n\n");

    get_system_information();
    

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
    // get transfer initialization costs
    timeval target_enter_data_costs = get_target_enter_data_costs(device_id);
    timeval target_exit_data_costs = get_target_exit_data_costs(device_id);
    timeval target_data_update_costs = get_target_data_update_costs(device_id);
    //get transfer times
    timeval H2D_1GB_costs = get_H2D_costs_1GB(device_id);
    double H2D_GB_s = 1 / (1*((int)H2D_1GB_costs.tv_sec)+ 0.000001*(long(H2D_1GB_costs.tv_usec)));
    timeval D2H_1GB_costs = get_D2H_costs_1GB(device_id);
    double D2H_GB_s = 1 / (1*D2H_1GB_costs.tv_sec + 0.000001*D2H_1GB_costs.tv_usec);
    

    // print stats
    printf("# DEVICE: %d\n", device_id);
    printf("# - processors: %d\n", num_processors);
    printf("# - teams: %d\n", max_teams);
    printf("# - target enter data costs: %ld.%06ld s\n", target_enter_data_costs.tv_sec, target_enter_data_costs.tv_usec);
    printf("# - target exit data costs:  %ld.%06ld s\n", target_exit_data_costs.tv_sec, target_exit_data_costs.tv_usec);
    printf("# - target data update costs: %ld.%06ld s\n", target_data_update_costs.tv_sec, target_data_update_costs.tv_usec);
    printf("# - Copy H2D 1GB costs: %ld.%06ld s\n", H2D_1GB_costs.tv_sec, H2D_1GB_costs.tv_usec);
    printf("#   - H2D: %f GB/s\n", H2D_GB_s);
    printf("# - Copy D2H 1GB costs: %ld.%06ld s\n", D2H_1GB_costs.tv_sec, D2H_1GB_costs.tv_usec);
    printf("#   - D2H: %f GB/s\n", D2H_GB_s);
    printf("\n");
}

timeval get_target_enter_data_costs(int device_id){
    // setup
    bool minimal_transfer_package = false;
    timeval before{}, after{};
    // measurement
    gettimeofday(&before, nullptr);
    #pragma omp target enter data map(to: minimal_transfer_package) device(device_id)
    gettimeofday(&after, nullptr);
    // cleanup
    #pragma omp target exit data map(delete: minimal_transfer_package) device(device_id)
    timeval result{};
    TimevalSubtract(before, after, result);
    return result;
}

timeval get_target_exit_data_costs(int device_id){
    // setup
    bool minimal_transfer_package = false;
    #pragma omp target enter data map(to: minimal_transfer_package) device(device_id)
    timeval before{}, after{};
    // measurement
    gettimeofday(&before, nullptr);
    #pragma omp target exit data map(delete: minimal_transfer_package) device(device_id)
    gettimeofday(&after, nullptr);
    // cleanup
    timeval result{};
    TimevalSubtract(before, after, result);
    return result;
}

timeval get_target_data_update_costs(int device_id){
    // setup
    bool minimal_transfer_package = false;
    #pragma omp target enter data map(to:minimal_transfer_package) device(device_id)
    minimal_transfer_package = true;
    timeval before{}, after{};
    // measurement
    gettimeofday(&before, nullptr);
    #pragma omp target update to(minimal_transfer_package) device(device_id)
    gettimeofday(&after, nullptr);
    // cleanup
    #pragma omp target exit data map(delete: minimal_transfer_package) device(device_id)
    timeval result{};
    TimevalSubtract(before, after, result);
    return result;
}

timeval get_H2D_costs_1GB(int device_id){
    // setup
    bool minimal_transfer_package = false;
    int32_t* array = new int32_t[250000000];  // size = 1GB
    
    #pragma omp target enter data map(to:minimal_transfer_package) map(alloc:array[0:250000000]) device(device_id)
    timeval before{}, after{};
    // measurement
    gettimeofday(&before, nullptr);
    #pragma omp target update to(array[0:250000000]) device(device_id)
    gettimeofday(&after, nullptr);
    // cleanup
    #pragma omp target exit data map(delete: minimal_transfer_package, array[0:250000000]) device(device_id)
    timeval result{};
    TimevalSubtract(before, after, result);
    return result;
}

timeval get_D2H_costs_1GB(int device_id){
    // setup
    bool minimal_transfer_package = false;
    int32_t* array = new int32_t[250000000];  // size = 1GB
    #pragma omp target enter data map(to:minimal_transfer_package) map(alloc:array[0:250000000]) device(device_id)
    #pragma omp target update to(array[0:250000000]) device(device_id)
    timeval before{}, after{};
    // measurement
    gettimeofday(&before, nullptr);
    #pragma omp target update from(array[0:250000000]) device(device_id)
    gettimeofday(&after, nullptr);
    // cleanup
    #pragma omp target exit data map(delete: minimal_transfer_package, array[0:250000000]) device(device_id)
    timeval result{};
    TimevalSubtract(before, after, result);
    return result;
}