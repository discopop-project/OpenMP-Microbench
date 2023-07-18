## compiler settings to enable openmp target offloading
CC=/home/lukasrothenberger/Software/clang-16_gpu/llvm-project/build/bin/clang
CXX=/home/lukasrothenberger/Software/clang-16_gpu/llvm-project/build/bin/clang++


OMP_FLAGS="-fopenmp=libomp -fopenmp-targets=nvptx64 --libomptarget-nvptx-bc-path=/home/lukasrothenberger/Software/clang-16_gpu/llvm-project/build/runtimes/runtimes-bins/openmp/libomptarget/libomptarget-nvptx-sm_61.bc -I/home/lukasrothenberger/Software/clang-16_gpu/llvm-project/build/runtimes/runtimes-bins/openmp/runtime/src -L/home/lukasrothenberger/Software/clang-16_gpu/llvm-project/build/runtimes/runtimes-bins/openmp/runtime/src --ptxas-path=/usr/local/cuda-11.4/bin/ptxas --cuda-path=/usr/local/cuda-11.4 --cuda-path-ignore-env"





## regular build + defined flags


cmake -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS} ${OMP_FLAGS} -fopenmo" -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${OMP_FLAGS} -fopenmp" . 
make

