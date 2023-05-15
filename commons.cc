#include <omp.h>
#include "commons.h"
#include "CLI11/CLI11.hpp"
#include "nlohmannJson/json.hpp"
#include <iomanip>
#include <iostream>
using json = nlohmann::json;

// The names of the different metrics for the JSON files
std::string METRIC_REFERENCE_TIME = "Reference time in us";
std::string METRIC_TEST_TIME = "Test time in us";
std::string METRIC_OVERHEAD = "Overhead time in us";

std::vector<unsigned int> TEST_REPETITIONS{};
std::vector<unsigned long long> NUMBER_OF_ITERATIONS{};
std::vector<unsigned int> NUMBER_OF_THREADS{};
std::vector<unsigned int> NUMBER_OF_TEAMS{};
std::vector<unsigned int> NUMBER_OF_GPU_THREADS{};
std::vector<unsigned long> AMOUNT_OF_WORKLOAD{};
std::vector<unsigned long long> ARRAY_SIZES{};
std::vector<unsigned int> NUMBER_OF_CHILD_NODES{};
std::vector<unsigned long long> TOTAL_AMOUNT_OF_TASKS{};

unsigned int DIRECTIVE_REPETITIONS;
bool CLAMP_LOW{false};
bool QUIET{false};
bool SAVE_FOR_EXTRAP;
bool EPCC{false};
bool EMPTY_PARALLEL_REGION{false};

json extrap_data;

/// @brief Runs the actual benchmark 'test' with the configuration given by 'datapoint'
/// Takes 'datapoint.repetitions' times the time and stores it in 'datapoint.time'
/// @param test the function to be benchmarked
/// @param datapoint the configuration for the benchmark, and the storage for the times
void Benchmark(void (&test)(const DataPoint&), DataPoint &datapoint);

int TimevalSubtract(timeval &before, timeval &after, timeval &result) {
    result.tv_sec = after.tv_sec - before.tv_sec;

    if ((result.tv_usec = after.tv_usec - before.tv_usec) < 0) {
        result.tv_usec += 1000000;
        result.tv_sec--; // borrow
    }

    return result.tv_sec < 0;
}

void ParseArgs(int argc, char **argv) {
    CLI::App app("OpenMP 4.5/5.0 Microbenchmark Suite");

    // Example config file is provided. Some parameters can be used as a vector "1 2 3" to evaluate each of these data points
    // This is possible as command line arguments or as a config file
    // Example config call: schedule_bench --config path/to/config.ini
    // Example config call: schedule_bench --config ../config.ini
    app.set_config("--config", "config.ini")->expected(0, 1);

    app.add_option("-R,--Repetitions", TEST_REPETITIONS, "Number of repetitions per microbenchmark (vector)(default: 5)");
    app.add_option("-T,--Threads", NUMBER_OF_THREADS, "Amount of total threads (vector)(default: max, max/2, max/4, ..., 1)");
    app.add_option("-I,--Iterations", NUMBER_OF_ITERATIONS, "Amount of iterations inside the constructs (vector)(default: 100)");
    app.add_option("-W,--Workload", AMOUNT_OF_WORKLOAD, "Workload in iterations inside the constructs (vector)(default: 2)");
    app.add_option("-D,--Directive", DIRECTIVE_REPETITIONS, "Amount of times the directives should be repeated(default: 1)");
    app.add_flag("--ClampLow", CLAMP_LOW, "Due to variance in measurements negative overheads are possible. This flag clamps overheads to values >=1.0");
    app.add_flag("-E,--ExtraP", SAVE_FOR_EXTRAP, "Saves the data as a json readable by ExtraP");
    app.add_flag("-Q,--Quiet", QUIET, "Disables the print to stdout");
    app.add_flag("--EPCC", EPCC, "[EXPERIMENTAL] Enables overhead calculation of EPCC");
    app.add_flag("--EmptyParallelRegion", EMPTY_PARALLEL_REGION, "Creates an empty parallel region with n threads before every benchmark to avoid measuring initial thread creation overhead");

    try {
        (app).parse((argc), (argv));
    }
    catch (const CLI::ParseError &e) {
        app.exit(e);
        exit(0);
    }

    sort(TEST_REPETITIONS.begin(), TEST_REPETITIONS.end(), std::greater<>());
    if (TEST_REPETITIONS.empty()) {
        TEST_REPETITIONS.push_back(5);
    }

    if (NUMBER_OF_THREADS.empty()) {
        for (int threads = omp_get_max_threads(); threads >= 1; threads = threads / 2) {
            NUMBER_OF_THREADS.push_back(threads);
        }
    }

    sort(NUMBER_OF_ITERATIONS.begin(), NUMBER_OF_ITERATIONS.end(), std::greater<>());
    if (NUMBER_OF_ITERATIONS.empty()) {
        NUMBER_OF_ITERATIONS.push_back(100);
    }

    sort(AMOUNT_OF_WORKLOAD.begin(), AMOUNT_OF_WORKLOAD.end(), std::greater<>());
    if (AMOUNT_OF_WORKLOAD.empty()) {
        AMOUNT_OF_WORKLOAD.push_back(2);
    }

    if (DIRECTIVE_REPETITIONS <= 0) {
        DIRECTIVE_REPETITIONS = 1;
    }
}

void Benchmark(void (&test)(const DataPoint&), DataPoint &datapoint) {

    std::vector<long double> result_vector;

    #pragma omp parallel num_threads(datapoint.threads)
    {
        // Most OpenMP implementations reuse threads once they are created,
        // so only the initial creation of threads is very costly.
        // Since this additional overhead is not incurred every time, we do not want to measure it.
        // This empty parallel region causes the threads to be created outside of the measured benchmark
    }

    for (int rep = 0; rep < datapoint.repetitions; rep++) {
        timeval before{}, after{};

        gettimeofday(&before, nullptr);

        test(datapoint);

        gettimeofday(&after, nullptr);

        timeval result{};
        TimevalSubtract(before, after, result);

        long double time = ((result.tv_sec * 1e6L) + result.tv_usec) / datapoint.directive;
        if (EPCC) {
            // EPCC CALCULATION
            result_vector.push_back((time * datapoint.threads) / (float(datapoint.iterations)));
        }
        else {
            result_vector.push_back(time);
        }

    }

    copy(result_vector.begin(), result_vector.end(), back_inserter(datapoint.time));
      
}

void Benchmark(const std::string &bench_name, const std::string &test_name,
               void (&test)(const DataPoint&), void (&ref)(const DataPoint&)) {

    // We set this one, so that OpenMP doesn't choose any number of threads <= num_threads
    // OpenMP will always choose the selected number of threads this way
    omp_set_dynamic(0);

    std::vector<DataPoint> reference_data;
    std::vector<DataPoint> test_data;
    std::vector<DataPoint> overhead_data;

    // We benchmark all possible permutations from the given parameters
    for (unsigned int repetitions : TEST_REPETITIONS) {
        for (unsigned long long iterations : NUMBER_OF_ITERATIONS) {
            for (unsigned long workload : AMOUNT_OF_WORKLOAD) {
                DataPoint single_reference_data{};
                single_reference_data.repetitions = repetitions;
                single_reference_data.iterations = iterations;
                single_reference_data.workload = workload;
                single_reference_data.directive = DIRECTIVE_REPETITIONS;
                single_reference_data.threads = 1;
                Benchmark(ref, single_reference_data);

                for (unsigned int threads: NUMBER_OF_THREADS) {
                    // exporting each reference measurement multiple times is not nice,
                    // but extrap needs the exact same amount of data to work properly
                    // TODO: we could use a separate "experiment" to avoid this
                    single_reference_data.threads = threads;
                    reference_data.push_back(single_reference_data);

                    // Preparation
                    DataPoint single_test_data;
                    DataPoint single_overhead_data;

                    if (EPCC) {
                        // EPCC calculates with iterations per thread, to have the same inputs as EPCC,
                        // we have to use the same calculations
                        iterations = iterations * threads;
                    }

                    single_test_data.repetitions = repetitions;
                    single_test_data.threads = threads;
                    single_test_data.iterations = iterations;
                    single_test_data.workload = workload;
                    single_test_data.directive = DIRECTIVE_REPETITIONS;

                    single_overhead_data.repetitions = repetitions;
                    single_overhead_data.threads = threads;
                    single_overhead_data.iterations = iterations;
                    single_overhead_data.workload = workload;
                    single_overhead_data.directive = DIRECTIVE_REPETITIONS;

                    // Benchmark section
                    Benchmark(test, single_test_data);

                    // Finishing up
                    if (EPCC) {
                        // Reverting EPCC iteration calculation, so that we display the results with our iterations again
                        single_test_data.iterations = iterations / threads;
                        single_overhead_data.iterations = iterations / threads;
                    }

                    test_data.push_back(single_test_data);

                    // Calculate Overhead := (T_parallel - (T_seq / N_threads))
                    for (int result = 0; result < single_test_data.time.size()/*<==>repetitions*/; result++) {
                        long double reference_time = single_reference_data.time.at(result);
                        long double test_time = single_test_data.time.at(result);
                        long double overhead = test_time - (reference_time / threads);

                        if(CLAMP_LOW && overhead <= 1.0){
                            overhead = 1.0;
                        }

                        single_overhead_data.time.push_back(overhead);
                    }

                    overhead_data.push_back(single_overhead_data);
                }
            }
        }
    }

    if (SAVE_FOR_EXTRAP) {
        SaveStatsForExtrap(bench_name, test_name, reference_data, METRIC_REFERENCE_TIME);
        SaveStatsForExtrap(bench_name, test_name, test_data, METRIC_TEST_TIME);
        SaveStatsForExtrap(bench_name, test_name, overhead_data, METRIC_OVERHEAD);
    }

    if (!QUIET) {
        PrintStats(test_name, overhead_data);
    }
}


void PrintStats(const std::string &test_name,
                const std::vector<DataPoint> &datapoints)
{
    std::cout << "Name of test: " << test_name << std::endl;

    std::cout << "Repetitions | Threads | Iterations | Workload in iterations | Overhead in us " << std::endl;

    for (auto data : datapoints) {
        sort(data.time.begin(), data.time.end());

        std::cout << std::setw(11) << data.repetitions
                  << " | " << std::setw(7) << data.threads
                  << " | " << std::setw(10) << data.iterations
                  << " | " << std::setw(22) << data.workload
                  << " | " << std::setw(14) << std::fixed << std::setprecision(4) << data.time.at(data.time.size()/2) << std::endl;
    }

    std::cout << "----------------------------------------------------------------------------" << std::endl;
}


void SaveStatsForExtrap(const std::string &bench_name,
                        const std::string &test_name,
                        const std::vector<DataPoint> &datapoints,
                        const std::string &metric_type){
    std::string file_name = {bench_name + "_runs.json"};
    std::ofstream file_to_write_to(file_name);

    json values_and_points = json::array({});
    json points_array = json::array({});
    json points;

    try {
        if (!extrap_data) {
            std::ifstream ifs(file_name);
            extrap_data = json::parse(ifs);
        }
    }
    catch (...) {
        //this is fine, since the file may be empty
    }
    values_and_points = extrap_data["measurements"][test_name][metric_type];

    for (const auto &data : datapoints)
    {
        points["point"] = {data.threads, data.workload, data.iterations};
        for (auto points_at_x : data.time){
            points_array += points_at_x;
        }

        points["values"] = points_array;
        values_and_points += points;
        points_array.clear();
        points.clear();
    }


    extrap_data["measurements"][test_name][metric_type] = values_and_points;
    extrap_data["parameters"] = {"Threads", "Workload", "Iterations"};

    file_to_write_to << std::setw(2) << extrap_data << std::endl;
}


float DelayFunction(unsigned int iteration, unsigned long workload) {
    float a = 0.0f;

    for (int i = 0; i < workload; i++) {
        a += (float) i;
    }
    if (a < 0) {
        printf("%f \n", a + (float) (iteration));
    }

    return a;
}

void ArrayDelayFunction(unsigned int iteration, unsigned long workload, float *a) {
    a[0] = 1.0;

    for (int i = 0; i < workload; i++) {
        a[0] += (float) i;
    }
    if (a[0] < 0) {
        printf("%f \n", a[0] + (float) (iteration));
    }
}


void ParseArgsGpu(int argc, char **argv) {
    
    unsigned int cpu_threads = -1;

    CLI::App app("OpenMP 4.5/5.0 Microbenchmark Suite");

    // Example config file is provided. Every parameter can be used as a vector "1 2 3", to evaluate each of these data points
    // The same is possible as Command line arguments, with the same layout as in config.ini
    // Example config call: gpu_offloading_bench --config path/to/config.ini
    // Example config call: gpu_offloading_bench --config ../config.ini
    app.set_config("--config", "configGPU.ini")->expected(0, 1);
    app.add_option("-R,--Repetitions", TEST_REPETITIONS, "Number of repetitions per Benchmark (vector)");
    app.add_option("--Threads", NUMBER_OF_GPU_THREADS, "Number of Threads per Team (vector)");
    app.add_option("--Teams", NUMBER_OF_TEAMS, "Number of Teams per Benchmark (vector)");
    app.add_option("--Reference", cpu_threads, "Number of CPU threads the GPU is compared to");
    app.add_option("-I,--Iterations", NUMBER_OF_ITERATIONS, "Amount of iterations inside the constructs (vector)");
    app.add_option("-N,--ArraySizes", ARRAY_SIZES, "Number of floats transferred to the GPU (vector)");
    app.add_option("-W,--Workload", AMOUNT_OF_WORKLOAD, "Workload in iterations inside the constructs (vector)");
    app.add_option("-D,--Directive", DIRECTIVE_REPETITIONS, "Amount of times the directives should be repeated");
    app.add_flag("-E,--ExtraP", SAVE_FOR_EXTRAP, "Saves the data as a Json readable by ExtraP");
    app.add_flag("-Q,--Quiet", QUIET, "Disables the print to stdout");


    try {
        (app).parse((argc), (argv));
    }
    catch (const CLI::ParseError &e) {
        app.exit(e);
        exit(0);
    }

    // Default Team and threads per Team Numbers are low to make sure most GPU can handle it
    sort(NUMBER_OF_GPU_THREADS.begin(), NUMBER_OF_GPU_THREADS.end());
    if (NUMBER_OF_GPU_THREADS.empty()) {
        NUMBER_OF_GPU_THREADS.push_back(10);
    }

    sort(NUMBER_OF_TEAMS.begin(), NUMBER_OF_TEAMS.end(), std::greater<>());
    if (NUMBER_OF_TEAMS.empty()) {
        NUMBER_OF_TEAMS.push_back(5);
    }

    // if 'cpu_threads' is not set, use all available threads
    if (cpu_threads <= 0) {
        NUMBER_OF_THREADS.push_back(omp_get_max_threads());
    }
    else {
        NUMBER_OF_THREADS.push_back(cpu_threads);
    }

    sort(NUMBER_OF_ITERATIONS.begin(), NUMBER_OF_ITERATIONS.end(), std::greater<>());
    if (NUMBER_OF_ITERATIONS.empty()) {
        NUMBER_OF_ITERATIONS.push_back(10);
    }

    sort(ARRAY_SIZES.begin(), ARRAY_SIZES.end(), std::greater<>());
    if (ARRAY_SIZES.empty()) {
        ARRAY_SIZES.push_back(100000);
    }

    sort(AMOUNT_OF_WORKLOAD.begin(), AMOUNT_OF_WORKLOAD.end(), std::greater<>());
    if (AMOUNT_OF_WORKLOAD.empty()) {
        AMOUNT_OF_WORKLOAD.push_back(10);
    }

    sort(TEST_REPETITIONS.begin(), TEST_REPETITIONS.end(), std::greater<>());
    if (TEST_REPETITIONS.empty()) {
       TEST_REPETITIONS.push_back(5);
    }

    if (DIRECTIVE_REPETITIONS <= 0) {
        DIRECTIVE_REPETITIONS = 1;
    }
}


void BenchmarkGpu(const std::string &bench_name,
                  const std::string &test_name,
                  void (&preprocessing)(const DataPoint&),
                  void (&test)(const DataPoint&),
                  void (&postprocessing)(),
                  void (&ref)(const DataPoint&)) {
    // We set this one, so that OpenMP doesn't choose any number of threads <= num_threads
    // OpenMP will always choose the selected number of threads this way
    omp_set_dynamic(0);
    std::vector<DataPoint> time_data;
    std::vector<DataPoint> overhead_data;
    
    for (unsigned int test_repetitions: TEST_REPETITIONS) {
        for (unsigned long long array_size: ARRAY_SIZES) {
            for (unsigned int iterations: NUMBER_OF_ITERATIONS) {
                for (unsigned int workload: AMOUNT_OF_WORKLOAD) {

                    DataPoint reference{};

                    reference.teams = 0;
                    reference.array_size = array_size;
                    reference.repetitions = test_repetitions;
                    reference.iterations = iterations;
                    reference.workload = workload;
                    reference.gpu_threads = 0;
                    reference.threads = NUMBER_OF_THREADS.at(0);
                    reference.directive = DIRECTIVE_REPETITIONS;

                    preprocessing(reference);
                    Benchmark(ref, reference);
                    postprocessing();

                    for (unsigned int teams: NUMBER_OF_TEAMS) {
                        for (unsigned int gpu_threads: NUMBER_OF_GPU_THREADS) {

                            // Preparation
                            DataPoint single_time_data;
                            DataPoint single_overhead_data;

                            single_time_data.teams = teams;
                            single_time_data.array_size = array_size;
                            single_time_data.repetitions = test_repetitions;
                            single_time_data.iterations = iterations;
                            single_time_data.workload = workload;
                            single_time_data.gpu_threads = gpu_threads;
                            single_time_data.threads=NUMBER_OF_THREADS.at(0);
                            single_time_data.directive = DIRECTIVE_REPETITIONS;

                            single_overhead_data.teams = teams;
                            single_overhead_data.array_size = array_size;
                            single_overhead_data.repetitions = test_repetitions;
                            single_overhead_data.iterations = iterations;
                            single_overhead_data.workload = workload;
                            single_overhead_data.gpu_threads = gpu_threads;
                            single_overhead_data.threads=NUMBER_OF_THREADS.at(0);
                            single_overhead_data.directive = DIRECTIVE_REPETITIONS;

                            // Benchmark section
                            preprocessing(single_time_data);
                            Benchmark(test, single_time_data);
                            postprocessing();

                            // Finishing up
                            time_data.push_back(single_time_data);

                            // T_overhead = T_gpu - T_cpu
                            for (int result = 0; result < single_time_data.time.size(); result++) {
                                long double ref_time = reference.time.at(result);
                                long double data = single_time_data.time.at(result);

                                single_overhead_data.time.push_back(data - ref_time);
                            }

                            overhead_data.push_back(single_overhead_data);

                            // to reduce the number of parameters, we treat team size and threads per team as different experiments
                            std::string specific_experiment_name = test_name + "_Teams_" + std::to_string(teams) +
                                                                        "_Threads_" +  std::to_string(gpu_threads);

                            if (SAVE_FOR_EXTRAP) {
                                SaveStatsForExtrapGpu(bench_name, specific_experiment_name, time_data, METRIC_TEST_TIME);
                                SaveStatsForExtrapGpu(bench_name, specific_experiment_name, overhead_data, METRIC_OVERHEAD);
                            }

                            if (!QUIET) {
                                PrintStatsGpu(specific_experiment_name, overhead_data);
                            }
                            overhead_data.clear();
                            time_data.clear();
                        }
                    }
                }
            }
        }
    }
}


void SaveStatsForExtrapGpu(const std::string &bench_name,
                           const std::string &test_name,
                           const std::vector<DataPoint> &datapoints,
                           const std::string &metric_type){
    std::string file_name = {bench_name + "_runs.json"};
    std::ofstream file_to_write_to(file_name);

    try {
        if (!extrap_data) {
            std::ifstream ifs(file_name);
            extrap_data = json::parse(ifs);
        }
    }
    catch (...) {
        //this is fine, since the file may be empty
    }

   
    json values_and_points = json::array({});
    json points_array = json::array({});
    json points;
    
    for (const auto &data : datapoints) {
        points["point"] = {data.array_size, data.iterations, data.workload};
        for (auto points_at_x : data.time){
            points_array += points_at_x;
        }
        
        points["values"] = points_array;
        values_and_points += points;
        points_array.clear();
        points.clear();
    }

    for (int i = 0; i < (extrap_data["measurements"][test_name][metric_type]).size(); i++)
    {
        values_and_points.push_back(extrap_data["measurements"][test_name][metric_type].at(i));
    }

    extrap_data["measurements"][test_name][metric_type] = values_and_points;
    extrap_data["parameters"] = {"Arraysize", "Iterations", "Workload"};

    file_to_write_to << std::setw(2) << extrap_data << std::endl;
}


void PrintStatsGpu(const std::string &test_name,
                   const std::vector<DataPoint> &datapoints)
{
    std::cout << "Name of test: " << test_name << std::endl;
    std::cout << "Compared to  " << datapoints.at(0).threads << " threads" << std::endl;
    // Overhead is not calculated yet, it is planned to have similar expressiveness like the EPCC calculation
    std::cout << "Teams  | Max Threads per Team |   Arraysize   | Iterations | Workload | Time in us " << std::endl;
    for (auto data : datapoints) {
        sort(data.time.begin(), data.time.end());

        std::cout <<  std::setw(6) << data.teams
                  << " | " << std::setw(20) << data.gpu_threads
                  << " | " << std::setw(13) << data.array_size
                  << " | " << std::setw(10) << data.iterations
                  << " | " << std::setw(8) << data.workload
        // median value of all repetitions
                  << " | " << std::setw(10) << data.time.at(data.time.size()/2) << std::endl;
    }
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
}


void ParseArgsTaskTree(int argc, char **argv)
{
    CLI::App app("OpenMP 4.5/5.0 Microbenchmark Suite");

    // Example config file is provided. Every parameter can be used as a vector "1 2 3", to evaluate each of these data points
    // The same is possible as Command line arguments, with the same layout as in config.ini
    // Example config call: tasktree_bench --config path/to/config.ini
    // Example config call: tasktree_bench --config ../config.ini
    app.set_config("--config", "configTaskTree.ini")->expected(0, 1);
    app.add_option("-R,--Repetitions", TEST_REPETITIONS, "Number of repetitions per Benchmark (vector)");
    app.add_option("-C,--Children", NUMBER_OF_CHILD_NODES, "Amount of Child Nodes getting created by one Node (vector)");
    app.add_option("-T,--Tasks", TOTAL_AMOUNT_OF_TASKS, "Amount of Tasks created to do work (vector)");
    app.add_option("-W,--Workload", AMOUNT_OF_WORKLOAD, "Workload done in Task (vector)");
    app.add_flag("-E,--ExtraP", SAVE_FOR_EXTRAP, "Saves the data as a Json readable by ExtraP");
    app.add_flag("-Q,--Quiet", QUIET, "Disables the print to stdout");

    try {
        (app).parse((argc), (argv));
    }
    catch (const CLI::ParseError &e) {
        app.exit(e);
        exit(0);
    }

    sort(TEST_REPETITIONS.begin(), TEST_REPETITIONS.end(), std::greater<>());
    if (TEST_REPETITIONS.empty()) {
        TEST_REPETITIONS.push_back(5);
    }

    if (NUMBER_OF_CHILD_NODES.empty()) {
        for (int threads = omp_get_max_threads(); threads > 1; threads = threads / 2) {
            NUMBER_OF_CHILD_NODES.push_back(threads);
        }
    }
    else {
        // We have to delete the case of just one child per node, because that would lead an infinite amount of nodes being created
        NUMBER_OF_CHILD_NODES.erase(std::remove(NUMBER_OF_CHILD_NODES.begin(), NUMBER_OF_CHILD_NODES.end(), 1), NUMBER_OF_CHILD_NODES.end());
    }
    sort(NUMBER_OF_CHILD_NODES.begin(), NUMBER_OF_CHILD_NODES.end());

    sort(TOTAL_AMOUNT_OF_TASKS.begin(), TOTAL_AMOUNT_OF_TASKS.end(), std::greater<>());
    if (TOTAL_AMOUNT_OF_TASKS.empty()) {
        TOTAL_AMOUNT_OF_TASKS.push_back(100);
    }

    sort(AMOUNT_OF_WORKLOAD.begin(), AMOUNT_OF_WORKLOAD.end(), std::greater<>());
    if (AMOUNT_OF_WORKLOAD.empty()) {
        AMOUNT_OF_WORKLOAD.push_back(2);
    }

    DIRECTIVE_REPETITIONS = 1;
}


void BenchmarkTaskTree(const std::string &bench_name,
                       const std::string &test_name,
                       void (&test)(const DataPoint&),
                       void (&ref)(const DataPoint&)) {
    // We set this one, so that OpenMP doesn't choose any number of threads <= num_threads
    // OpenMP will always choose the selected number of threads this way
    omp_set_dynamic(0);
    std::vector<DataPoint> overhead_data;
    std::vector<DataPoint> time_data;

    for (unsigned int test_repetitions: TEST_REPETITIONS) {
        for (unsigned long long tasks: TOTAL_AMOUNT_OF_TASKS) {
            for (unsigned long long workload: AMOUNT_OF_WORKLOAD) {
                DataPoint reference{};
                reference.repetitions = test_repetitions;
                reference.tasks = tasks;
                reference.child_nodes = 0;
                reference.workload = workload;
                reference.directive = DIRECTIVE_REPETITIONS;

                Benchmark(ref, reference);

                for (unsigned long long child_nodes: NUMBER_OF_CHILD_NODES) {

                    // Preparation
                    DataPoint single_time_data;
                    DataPoint single_overhead_data;

                    single_time_data.repetitions = test_repetitions;
                    single_time_data.tasks = tasks;
                    single_time_data.child_nodes = child_nodes;
                    single_time_data.workload = workload;
                    single_time_data.directive = DIRECTIVE_REPETITIONS;

                    single_overhead_data.repetitions = test_repetitions;
                    single_overhead_data.tasks = tasks;
                    single_overhead_data.child_nodes = child_nodes;
                    single_overhead_data.workload = workload;
                    single_overhead_data.directive = DIRECTIVE_REPETITIONS;

                    // Benchmark section
                    Benchmark(test, single_time_data);

                    // Finishing up
                    time_data.push_back(single_time_data);

                    // T_overhead = T_tasktree - T_taskloop
                    for (int result = 0; result < single_time_data.time.size(); result++) {
                        long double ref_time = reference.time.at(result);
                        long double data = single_time_data.time.at(result);

                        single_overhead_data.time.push_back(data - ref_time);
                    }

                    overhead_data.push_back(single_overhead_data);
                }
            }
        }
    }

    if (SAVE_FOR_EXTRAP) {
        SaveStatsForExtrapTaskTree(bench_name, test_name, time_data, METRIC_TEST_TIME);
        SaveStatsForExtrapTaskTree(bench_name, test_name, overhead_data, METRIC_OVERHEAD);
    }

    if (!QUIET) {
        PrintStatsTaskTree(test_name, overhead_data);
    }
}



void PrintStatsTaskTree(const std::string &test_name,
                        const std::vector<DataPoint> &datapoints)
{
    std::cout << "Name of test: " << test_name << std::endl;

    std::cout << "Repetitions | Branches | Number of Tasks to create | Workload foreach Task | Overhead in us " << std::endl;

    for (auto data : datapoints)
    {
        sort(data.time.begin(), data.time.end());

        std::cout << std::setw(11) << data.repetitions
                  << " | " << std::setw(8) << data.child_nodes
                  << " | " << std::setw(25) << data.tasks
                  << " | " << std::setw(21) << data.workload
                  << " | " << std::setw(14) << data.time.at(data.time.size()/2) << std::endl;
    }
    std::cout << "----------------------------------------------------------------------------" << std::endl;
}


void SaveStatsForExtrapTaskTree(const std::string &bench_name,
                                const std::string &test_name,
                                const std::vector<DataPoint> &datapoints,
                                const std::string& metric_type){
    std::string file_name = {bench_name + "_runs.json"};
    std::ofstream file_to_write_to(file_name);
   
    json values_and_points = json::array({});
    json points_array = json::array({});
    json points;

    try {
        if (!extrap_data) {
            std::ifstream ifs(file_name);
            extrap_data = json::parse(ifs);
        }
    }
    catch (...) {
        //this is fine, since the file may be empty
    }
    
    for (const auto &data : datapoints)
    {
        points["point"] = {data.child_nodes, data.workload, data.tasks};
        for (auto points_at_x : data.time){
            points_array += points_at_x;
        }

        points["values"] = points_array;
        values_and_points += points;
        points_array.clear();
        points.clear();
    }


    extrap_data["measurements"][test_name][metric_type] = values_and_points;
    extrap_data["parameters"] = {"Branches", "Workload", "Number of Tasks"};

    file_to_write_to << std::setw(2) << extrap_data << std::endl;
}

int ReturnFalse() {
    return 0;
}

int ReturnTrue() {
    return 1;
}

void RemoveBench(std::string &bench_name) {
    std::remove((bench_name + "_runs.json").c_str());
}

void PrintCompilerVersion() {
    if(!QUIET) {
        #ifdef __clang__
        printf("using %s %d.%d.%d\n", "clang", __clang_major__, __clang_minor__, __clang_patchlevel__);
        #else
        printf("using %s %d.%d.%d\n", "gcc", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
        #endif
    }
}