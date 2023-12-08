#ifndef PPT_P4_SYSTEM_H
#define PPT_P4_SYSTEM_H

#include "commons.h"

void test_system();
void test_gpu_execution(int device_id);
void get_system_information();
void get_device_information(int device_id);
void get_host_information(int device_id);

timeval get_target_enter_data_costs(int device_id);
timeval get_target_exit_data_costs(int device_id);
timeval get_target_data_update_costs(int device_id);

#endif //PPT_P4_SYSTEM_H
