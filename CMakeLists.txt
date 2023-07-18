cmake_minimum_required(VERSION 3.23)
project(ppt_p4)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

include(FindOpenMP)

function(doa_all_bench_gen array_size)
    add_executable(doall_bench_${array_size} doall_bench.cc commons.cc)
    target_compile_definitions(doall_bench_${array_size} PRIVATE ARRAY_SIZE=${array_size})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(doall_bench_${array_size} PUBLIC OpenMP::OpenMP_CXX)
    endif()
endfunction(doa_all_bench_gen)

#foreach (array_size 1 4 16 64 256 1024 4096 16384 65536 262144)
foreach (array_size 1 1024)
    doa_all_bench_gen(${array_size})
endforeach ()

add_executable(reduction_bench
        reduction_bench.cc
        commons.cc)

#add_executable(task_bench
#        task_bench.cc
#        commons.cc)

#add_executable(schedule_bench
#        schedule_bench.cc
#        commons.cc)

#add_executable(sync_bench
#        sync_bench.cc
#        commons.cc)

add_executable(gpuoffloading_bench
        gpuoffloading_bench.cc
        commons.cc)

#add_executable(tasktree_bench
#        tasktree_bench.cc
#        commons.cc)

if(OPENMP_FOUND)
    target_link_libraries(reduction_bench PUBLIC OpenMP::OpenMP_CXX)
#    target_link_libraries(task_bench PUBLIC OpenMP::OpenMP_CXX)
#    target_link_libraries(schedule_bench PUBLIC OpenMP::OpenMP_CXX)
#    target_link_libraries(sync_bench PUBLIC OpenMP::OpenMP_CXX)
#    target_link_libraries(gpuoffloading_bench PUBLIC OpenMP::OpenMP_CXX)
#    target_link_libraries(tasktree_bench PUBLIC OpenMP::OpenMP_CXX)
endif()
