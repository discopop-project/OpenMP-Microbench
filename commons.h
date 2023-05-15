#ifndef PPT_P4_COMMONS_H
#define PPT_P4_COMMONS_H

#include <sys/time.h>
#include <string>
#include <vector>


/// @brief Used as a data storage, and microbenchmark configuration
struct DataPoint {
public:
    unsigned long long iterations;
    unsigned int threads;
    unsigned long workload;
    std::vector<long double> time;
    unsigned int repetitions;
    unsigned int directive;

    /// Only used in TaskTree
    unsigned int tasks;
    unsigned int child_nodes;

    /// Only used in GPUOffloading
    unsigned int teams;
    unsigned long long array_size;
    unsigned int gpu_threads;
};

/// @brief The number of times each test should get repeated, for median taking
extern std::vector<unsigned int> TEST_REPETITIONS;

/// @brief The number of loop iterations in each microbenchmark
extern std::vector<unsigned long long> NUMBER_OF_ITERATIONS;

/// @brief The number of threads used in each microbenchmark
extern std::vector<unsigned int> NUMBER_OF_THREADS;

/// @brief The number of calculations in each loop iteration and task
extern std::vector<unsigned long> AMOUNT_OF_WORKLOAD;

/// @brief The sizes of the arrays copied to the target (GPU)
extern std::vector<unsigned long long> ARRAY_SIZES;

/// @brief The number of teams in use on the GPU
extern std::vector<unsigned int> NUMBER_OF_TEAMS;

/// @brief The number of child nodes for every node for the task tree benchmark
extern std::vector<unsigned int> NUMBER_OF_CHILD_NODES;

/// @brief The number of total tasks to be created for the task tree benchmark
extern std::vector<unsigned long long> TOTAL_AMOUNT_OF_TASKS;

/// @brief The number of times each directive gets repeated, for mean taking
extern unsigned int DIRECTIVE_REPETITIONS;

/// @brief Sets whether the medians should get printed to the stdout
extern bool QUIET;

/// @brief Sets whether all measurements should get saved in a JSON file for ExtraP
extern bool SAVE_FOR_EXTRAP;

/// @brief [EXPERIMENTAL] Overhead calculation in a similar style to EPCC, for comparison
extern bool EPCC;

/// @brief Calculates the difference between before and after time measurements
/// @param before the earlier time
/// @param after the later time
/// @param result the difference of before and after
int TimevalSubtract(timeval &before, timeval &after, timeval &result);

/// @brief Parsing of the commandline arguments and/or config file.
/// The arguments can be seen in the readme file
void ParseArgs(int argc, char **argv);

/// @brief The same as ParseArgs, but with different arguments, for the GPU
/// @ref ParseArgs
void ParseArgsGpu(int argc , char **argv);

/// @brief The same as ParseArgs, but with different arguments, for the task tree creation
/// @ref ParseArgs
void ParseArgsTaskTree(int argc , char **argv);

/// @brief These benchmark methods are getting called by the microbenchmarks itself
/// @param bench_name the bench_name of the microbenchmark
/// @param test_name the bench_name of the individual test
/// @param test the reference to the function to get benchmarked
/// @param ref the reference to the reference function, typically a serial implementation of the same code
void Benchmark(const std::string &bench_name, const std::string &test_name, void (&test)(const DataPoint&),
               void (&ref)(const DataPoint&));

/// @brief These benchmark methods are getting called by the microbenchmarks itself
/// @param bench_name the name of the microbenchmark
/// @param test_name the name of the individual test
/// @param preprocessing called before every benchmark, including the reference
/// @param test the reference to the function to get benchmarked
/// @param preprocessing called after every benchmark, including the reference
/// @param ref the reference to the reference function, typically a serial implementation of the same code
void BenchmarkGpu(const std::string &bench_name, const std::string &test_name, void (&preprocessing)(const DataPoint&),
                  void (&test)(const DataPoint&), void (&postprocessing)(), void (&ref)(const DataPoint&));

/// @brief These benchmark methods are getting called by the microbenchmarks itself
/// @param bench_name the name of the microbenchmark
/// @param test_name the name of the individual test
/// @param test the reference to the function to get benchmarked
/// @param ref the reference to the reference function, typically a serial implementation of the same code
void BenchmarkTaskTree(const std::string &bench_name, const std::string &test_name,
                       void (&test)(const DataPoint&), void (&ref)(const DataPoint&));


/// @brief Prints the median of each benchmark
/// @param test_name the name of each individual test
/// @param datapoints the measurements that are getting collected by the Benchmark function
void PrintStats(const std::string &test_name, const std::vector<DataPoint> &datapoints);

/// @brief Prints the median of each benchmark
/// @param test_name the name of each individual test
/// @param datapoints the measurements that are getting collected by the Benchmark function
void PrintStatsGpu(const std::string &test_name, const std::vector<DataPoint> &datapoints);

/// @brief Prints the median of each benchmark
/// @param test_name the name of each individual test
/// @param datapoints the measurements that are getting collected by the Benchmark function
void PrintStatsTaskTree(const std::string &test_name, const std::vector<DataPoint> &datapoints);

/// @brief Saves the statistics of each microbenchmark as a JSON file.
/// The exact structure of the JSON files can be read here:
/// @sa https://github.com/extra-p/extrap/blob/master/docs/file-formats.md#json-format
/// @param bench_name the name of the microbenchmarks
/// @param test_name the name of each individual test
/// @param datapoints the measurements that are getting collected by the Benchmark function
/// @param metric_type describes the name of the metric (Overhead or Time)
void SaveStatsForExtrap(const std::string &bench_name, const std::string &test_name,
                        const std::vector<DataPoint> &datapoints, const std::string& metric_type);

/// @brief Saves the statistics of each microbenchmark as a JSON file.
/// The exact structure of the JSON files can be read here:
/// @sa https://github.com/extra-p/extrap/blob/master/docs/file-formats.md#json-format
/// @param bench_name the name of the microbenchmarks
/// @param test_name the name of each individual test
/// @param datapoints the measurements that are getting collected by the Benchmark function
/// @param metric_type describes the name of the metric (Overhead or Time)
void SaveStatsForExtrapGpu(const std::string &bench_name, const std::string &test_name,
                           const std::vector<DataPoint> &datapoints, const std::string& metric_type);

/// @brief Saves the statistics of each microbenchmark as a JSON file.
/// The exact structure of the JSON files can be read here:
/// @sa https://github.com/extra-p/extrap/blob/master/docs/file-formats.md#json-format
/// @param bench_name the name of the microbenchmarks
/// @param test_name the name of each individual test
/// @param datapoints the measurements that are getting collected by the Benchmark function
/// @param metric_type describes the name of the metric (Overhead or Time)
void SaveStatsForExtrapTaskTree(const std::string &bench_name, const std::string &test_name,
                                const std::vector<DataPoint> &datapoints, const std::string& metric_type);

/// @brief Performs a predefined number of calculations and returns the result
/// @param iteration the current iteration of the surrounding loop construct (optional)
/// @param workload the number of calculations that should get performed
float DelayFunction(unsigned int iteration, unsigned long workload);


/// @brief Performs a predefined number of calculations and returns the result
/// @param iteration the current iteration of the surrounding loop construct (optional)
/// @param workload the number of calculations that should get performed
/// @param a the array that should get used for computations
void ArrayDelayFunction(unsigned int iteration, unsigned long workload, float *a);

/// @brief Returns false, used for conditional in OpenMP constructs.
/// @returns false
int ReturnFalse();

/// @brief Returns true, used for conditional in OpenMP constructs.
/// @returns false
int ReturnTrue();

/// @brief Removes the stored JSON file for the corresponding benchmark,
/// in order to avoid mixed results, and an unclean measurement collection
void RemoveBench(std::string &bench_name);

#endif //PPT_P4_COMMONS_H
