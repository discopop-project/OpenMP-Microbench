#include "system_bench.h"
#include "commons.h"
#include <omp.h>
#include <iostream>
#include <unistd.h>
#include "nlohmannJson/json.hpp"
#include <fstream>

using json = nlohmann::json;

std::string bench_name = "SYSTEM";

int main(int argc, char **argv){

    PrintCompilerVersion();
    printf("\n");

    if (SAVE_FOR_EXTRAP) {
        RemoveBench(bench_name);
    }

    test_system();

    printf("\n#################\n\n");

    json system_information;

    get_system_information(system_information);


    // export system information to file
    std::string file_name = "system_information.json";
    std::ofstream file_stream(file_name, std::ofstream::out);
    file_stream << std::setw(2) << system_information << std::endl;

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

void get_system_information(json &system_information){
    printf("# SYSTEM:\n");
    printf("# - devices: %d\n", omp_get_num_devices());
    printf("# - host_device: %d\n", omp_get_initial_device());
    printf("\n");

    // prepare system_information json
    system_information["devices"] = json::array({});
    system_information["host_device"] = omp_get_initial_device();

    get_host_information(omp_get_initial_device(), system_information);

    for(int device_id = 0; device_id < omp_get_num_devices(); device_id++){
        get_device_information(device_id, system_information);
    }
}

void get_host_information(int device_id, json &system_information){
    int num_processors = -1;
    int max_threads = -1;
    long frequency = 3000000000;
    // gather stats
    #pragma omp parallel
    #pragma omp single
    {
        num_processors = omp_get_num_procs();
        max_threads = omp_get_max_threads();
    }
    // get benchmark execution time
    timeval seq_execution_time = get_sequential_computation_time();
    double seq_execution_time_s = (1*seq_execution_time.tv_sec + 0.000001*seq_execution_time.tv_usec);
    timeval doall_execution_time = get_doall_computation_time();
    double doall_execution_time_s = (1*doall_execution_time.tv_sec + 0.000001*doall_execution_time.tv_usec);
    double doall_speedup = seq_execution_time_s / doall_execution_time_s;
    timeval doall_init_costs = get_doall_init_costs();
    double doall_init_costs_s = (1*((int)doall_init_costs.tv_sec)+ 0.000001*(long(doall_init_costs.tv_usec)));
    // print stats
    printf("# HOST: %d\n", device_id);
    printf("# - processors: %d\n", num_processors);
    printf("# - threads: %d\n", max_threads);
    printf("# - frequency (hardcoded): %ld Hz\n", frequency);
    printf("# - doall init costs: %f s\n", doall_init_costs_s);
    printf("# - sequential execution time: %f\n", seq_execution_time_s);
    printf("# - doall execution time: %f\n", doall_execution_time_s);
    printf("#   - speedup: %f\n", doall_speedup);
    printf("\n");
    // add information to json
    json host_device;
    host_device["device_id"] = device_id;
    host_device["processors"] = num_processors;
    host_device["threads"] = max_threads;
    host_device["frequency"] = frequency;
    json compute_init_costs;
    compute_init_costs["doall"] = doall_init_costs_s;
    host_device["compute_init_costs"] = compute_init_costs;
    system_information["devices"] += host_device;
}

void get_device_information(int device_id, json &system_information){
    int num_processors = -1;
    int max_threads = -1;
    int max_teams = -1;
    long frequency = 512000000;
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
    // get computation initialization costs
    timeval target_tdpf_init_costs = get_target_teams_distribute_parallel_for_init_costs(device_id);
    double target_tdpf_init_costs_s = (1*target_tdpf_init_costs.tv_sec + 0.000001*target_tdpf_init_costs.tv_usec);
    // get transfer initialization costs
    timeval target_enter_data_init_costs = get_target_enter_data_init_costs(device_id);
    double target_enter_data_init_costs_s = (1*target_enter_data_init_costs.tv_sec + 0.000001*target_enter_data_init_costs.tv_usec);
    timeval target_exit_data_init_costs = get_target_exit_data_init_costs(device_id);
    double target_exit_data_init_costs_s = (1*target_exit_data_init_costs.tv_sec + 0.000001*target_exit_data_init_costs.tv_usec);
    timeval target_data_update_init_costs = get_target_data_update_init_costs(device_id);
    double target_data_update_init_costs_s = (1*target_data_update_init_costs.tv_sec + 0.000001*target_data_update_init_costs.tv_usec);
    //get transfer times
    timeval H2D_1GB_costs = get_H2D_costs_1GB(device_id);
    double H2D_GBps = 1 / (1*((int)H2D_1GB_costs.tv_sec)+ 0.000001*(long(H2D_1GB_costs.tv_usec)));
    timeval D2H_1GB_costs = get_D2H_costs_1GB(device_id);
    double D2H_GBps = 1 / (1*D2H_1GB_costs.tv_sec + 0.000001*D2H_1GB_costs.tv_usec);
    // get benchmark execution time
    timeval device_execution_time = get_device_computation_time(device_id);
    double device_execution_time_s = (1*device_execution_time.tv_sec + 0.000001*device_execution_time.tv_usec);

    // print stats
    printf("# DEVICE: %d\n", device_id);
    printf("# - processors: %d\n", num_processors);
    printf("# - teams: %d\n", max_teams);
    printf("# - frequency (hardcoded): %ld Hz\n", frequency);
    printf("# - target teams distribute parallel for init costs: %ld.%06ld s\n", target_tdpf_init_costs.tv_sec, target_tdpf_init_costs.tv_usec);
    printf("# - target enter data init costs: %ld.%06ld s\n", target_enter_data_init_costs.tv_sec, target_enter_data_init_costs.tv_usec);
    printf("# - target exit data init costs:  %ld.%06ld s\n", target_exit_data_init_costs.tv_sec, target_exit_data_init_costs.tv_usec);
    printf("# - target data update init costs: %ld.%06ld s\n", target_data_update_init_costs.tv_sec, target_data_update_init_costs.tv_usec);
    printf("# - Copy H2D 1GB costs: %ld.%06ld s\n", H2D_1GB_costs.tv_sec, H2D_1GB_costs.tv_usec);
    printf("#   - H2D: %f GB/s\n", H2D_GBps);
    printf("# - Copy D2H 1GB costs: %ld.%06ld s\n", D2H_1GB_costs.tv_sec, D2H_1GB_costs.tv_usec);
    printf("#   - D2H: %f GB/s\n", D2H_GBps);
    printf("# - execution time: %f\n", device_execution_time_s);
    printf("\n");

    // add information to json
    json device;
    device["device_id"] = device_id;
    device["processors"] = num_processors;
    device["threads"] = max_teams;
    device["teams"] = max_teams;
    device["frequency"] = frequency;
    json compute_init_costs;
    compute_init_costs["target_teams_distribute_parallel_for"] = target_tdpf_init_costs_s;
    device["compute_init_costs"] = compute_init_costs;
    json transfer_init_costs;
    transfer_init_costs["target_enter_data"] = target_enter_data_init_costs_s;
    transfer_init_costs["target_exit_data"] = target_exit_data_init_costs_s;
    transfer_init_costs["target_data_update"] = target_data_update_init_costs_s;
    device["transfer_init_costs"] = transfer_init_costs;
    json transfer_speeds;
    transfer_speeds["H2D"] = H2D_GBps;
    transfer_speeds["D2H"] = D2H_GBps;
    device["transfer_speeds"] = transfer_speeds;
    system_information["devices"] += device;
}

timeval get_doall_init_costs(){
    // setup
    int iterations = 1000000;
    int workload = 1;
    timeval before{}, after{};
    // measurement
    gettimeofday(&before, nullptr);
    #pragma omp parallel for
    for (int i = 0; i < iterations; i++) {
            DELAY(workload, i);
    }
    gettimeofday(&after, nullptr);
    // cleanup
    timeval result{};
    TimevalSubtract(before, after, result);
    return result;
}

timeval get_target_teams_distribute_parallel_for_init_costs(int device_id){
    // setup
    int iterations = 1000000;
    int workload = 1;
    timeval before{}, after{};
    // measurement
    gettimeofday(&before, nullptr);
    #pragma omp target teams distribute parallel for device(device_id)
    for (int i = 0; i < iterations; i++) {
            DELAY(workload, i);
    }
    gettimeofday(&after, nullptr);
    // cleanup
    timeval result{};
    TimevalSubtract(before, after, result);
    return result;
}

timeval get_target_enter_data_init_costs(int device_id){
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

timeval get_target_exit_data_init_costs(int device_id){
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

timeval get_target_data_update_init_costs(int device_id){
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

timeval get_sequential_computation_time(){
    // setup
    int iterations = 1000000;
    int workload = 1000;
    timeval before{}, after{};
    // measurement
    gettimeofday(&before, nullptr);
    for (int i = 0; i < iterations; i++) {
            DELAY(workload, i);
    }
    gettimeofday(&after, nullptr);
    // cleanup
    timeval result{};
    TimevalSubtract(before, after, result);
    return result;
}

timeval get_doall_computation_time(){
    // setup
    int iterations = 1000000;
    int workload = 1000;
    timeval before{}, after{};
    // measurement
    gettimeofday(&before, nullptr);
    #pragma omp parallel for
    for (int i = 0; i < iterations; i++) {
            DELAY(workload, i);
    }
    gettimeofday(&after, nullptr);
    // cleanup
    timeval result{};
    TimevalSubtract(before, after, result);
    return result;
}

timeval get_device_computation_time(int device_id){
    // setup
    int iterations = 1000000;
    int workload = 1000;
    timeval before{}, after{};
    // measurement
    gettimeofday(&before, nullptr);
    #pragma omp target teams distribute parallel for device(device_id)
    for (int i = 0; i < iterations; i++) {
            DELAY(workload, i);
    }
    gettimeofday(&after, nullptr);
    // cleanup
    timeval result{};
    TimevalSubtract(before, after, result);
    return result;
}

