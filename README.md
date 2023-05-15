# Overhead measuring for OpenMP 5.2

## Description

## How to use
1. Build the repository with 
    ```bash 
    cmake .
    make all
    ```
2. Edit the config files
    ```
    configGPU.ini           for GPUOffloading
    configTaskTree.ini      for TaskTree 
    config.ini              for everything else
    ```
3. Run the benchmark, the config file is automatically detected, if it's in the same directory
4. Look at the output with ExtraP or in the stdout

## How to set a different compiler
This make file is automatically created by cmake with the CMakeList.txt, which uses the systems standard compiler.
If you want to set a different compiler, you can define it and build the project the following way:

```bash
export CXX=/path/to/compiler cmake /path/to/this/repo
make all
```

## How to compile for GPU
If the compiler is set up correctly with all functionality needed for offloading code for your GPU, then the compilation process is no different from described above.

## Parameter Description
### config.ini
| Parameter       | Description                                                                                                                                                   |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Repetitions     | Space separated list of the number repetitions of each microbenchmark                                                                                         |
| Threads         | Space separated list of the number of threads to be used for benchmarking                                                                                     |
| Iterations      | Space separated list of the amount of times the workload should be repeated (for taskbench this is the amount of tasks)                                       |
| Workload        | Space separated list of the amount of workload in each loop iteration                                                                                         |
| Directive       | The amount of times a directive should be repeated                                                                                                            |
| ExtraP          | Whether the output should be saved as a ExtraP-readable JSON format                                                                                           |
| Quiet           | If set to true the results wont get printed to stdout                                                                                                         |
| EPCC            | [EXPERIMENTAL] Enables EPCC-style overhead calculations. <br/> BEWARE: Some microbenchmarks behave differently than EPCC and therefore have different results |
| ClampLow        | Clamp low/negative overheads to 1.0 (ExtraP does not like values less than 1)                                                                                 |
| EmptyParallelRegion | Add an empty parallel region before the benchmark is run. <br/>(Most OpenMP implementations have more overhead when creating threads for the first time.) |


### configGPU.ini
| Parameter   | Description                                                                             |
|-------------|-----------------------------------------------------------------------------------------|
| Repetitions | Space separated list of the number of repetitions of each microbenchmark                |
| Threads     | Space separated list of the amount of threads per team on the GPU                       |
| Teams       | Space separated list of the amount of teams on the GPU                                  |
| Iterations  | Space separated list of the amount of times the workload should be repeated  on the GPU |
| ArraySizes  | Space separated list of the amount of floats that should be transferred to the GPU      |
| Workload    | Space separated list of the amount of workload in each iteration                        |
| Reference   | The amount of CPU threads the GPU overhead is calculated with (default = max threads)   |
| Directive   | The amount of times a directive should be repeated                                      |
| ExtraP      | Whether the output should be saved as a ExtraP-readable JSON format                     |
| Quiet       | If set to true the results wont get printed to stdout                                   |
### configTaskTree.ini
| Parameter   | Description                                                              |
|-------------|--------------------------------------------------------------------------|
| Repetitions | Space separated list of the number of repetitions of each microbenchmark |
| Children    | Space separated list of the amount of children that each node creates    |
| Tasks       | Space separated list of the total amount of tasks that should be created |
| Workload    | Space separated list of the amount of workload for each task             |
| ExtraP      | Whether the output should be saved as a ExtraP-readable JSON format      |
| Quiet       | If set to true the results wont get printed to stdout                    |

## Example Configuration
### config.ini
```ini
Repetitions=10
Threads=16 12 8 4 2
Iterations=100 1000 2000 4000 5000
Workload=10 50 100 200 500 1000
Directive=5
ExtraP=true
Quiet=false
EPCC=false
```
### configGPU.ini
```ini
Repetitions=10
Threads=10
Teams=5
Iterations=10
ArraySizes=1000
Workload=100
Reference=8
ExtraP=true
Quiet=true
```
### configTaskTree.ini
```ini
Repetitions=10 
Children=4 2 1
Tasks=10 20 1000
Workload=1 20 50
ExtraP=true
Quiet=true
```

