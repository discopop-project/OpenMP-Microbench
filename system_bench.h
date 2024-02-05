#ifndef PPT_P4_SYSTEM_H
#define PPT_P4_SYSTEM_H

#include "commons.h"
#include "nlohmannJson/json.hpp"

using json = nlohmann::json;

void test_system();
void test_gpu_execution(int device_id);
void get_system_information(json &system_information);
void get_device_information(int device_id, json &system_information);
void get_host_information(int device_id, json &system_information);

void prepare_execution_environment();

timeval get_doall_init_costs();
timeval get_target_teams_distribute_parallel_for_init_costs(int device_id);

timeval get_target_enter_data_init_costs(int device_id);
timeval get_target_exit_data_init_costs(int device_id);
timeval get_target_data_update_init_costs(int device_id);

timeval get_H2D_costs_1GB(int device_id);
timeval get_D2H_costs_1GB(int device_id);

timeval get_sequential_computation_time();
timeval get_doall_computation_time();
timeval get_device_computation_time(int device_id);

#endif //PPT_P4_SYSTEM_H
