/** EQUISPACED GRID MULTISCALE PARALLEL APPROXIMATION **/
/** brief explanation on ReadMe **/

// standard libraries
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

// cuda libraries
#include <cuda.h>
#include <cuda_runtime.h>

#include "macros.h" // macros with fixed constants (es point/values dim)
#include "multiscale_structure.cuh" // structure and kd construction
#include "iterative_parallel_methods.cuh" // solution of the approximation problem
#include "wendland_functions.h" // wendland functions for result evaluation

#define DATA_FILE "/100M_evaluated_domain.csv"

int main(int argc, char * argv[]){
    /// Expecting parameters are (ordered): Number of levels, mu and nu.
    // arguments definition
    unsigned int number_of_levels;
    double mu;
    double nu;
    // arguments handling and allocation
    if (argc > 4 || argc == 1){
        fprintf(stderr, "Expecting 1 to 3 parameters, had %d.\n", argc-1);
        return EXIT_FAILURE;
    } else if (argc == 2){
        number_of_levels = atoi(argv[1]);
        mu = 0.5;
        nu = 4;
    } else if (argc == 3){
        number_of_levels = atoi(argv[1]);
        mu = atof(argv[2]);
        nu = 4;
    } else {
        number_of_levels = atoi(argv[1]);
        mu = atof(argv[2]);
        nu = atof(argv[3]);
    }

    /// device setup, gathering basic information
    int deviceId, nSMs;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&nSMs, cudaDevAttrMultiProcessorCount, deviceId);
    // cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, deviceId); we assumed that warp_size = 32.
    /// parameters of the gpu setup
    printf("Running with %d levels, mu = %f, nu = %f\n", number_of_levels, mu, nu);

    /// data allocation
    // We plan to store data in the data in a tree structure level-wise. For access convenience we will use 2 structures to store the data.
    // For the tree rearrangement (level-wise) the data will be store first dimensional-wise then level-wise:
    // i.e. (for 3d data on L levels where each level has N(level) points) x_11, x_12, ..., x_1N(1), y_11, y_12, ..., y_1N(1), z_11, z_21, ..., z_1N(1), x_21, x_22, ..., x_2N(2), y_21, y_22, ..., y_2N(2), z_21, z_22, ..., z_2N(2), ..., x_L1, x_L2, ..., x_LN(L), y_L1, y_L2, ..., y_LN(L), z_L1, ..., z_LN(L)
    // For the system solution (intra-level)the data will be store first level-wise then dimensional-wise: (this will be applied inside the multiscale_free_exact_parallel_interpolation function)
    // i.e. (for 3d data on L levels where each level has N(level) points) x_11, x_12, ..., x_1N(1), x_21, x_22, ..., x_2N(2), ..., x_L1, x_L2, ..., x_LN(L), y_11, y_12, ..., y_1N(1), y_21, y_22, ..., y_2N(2), ..., y_L1, y_L2, ..., y_LN(L), z_11, z_21, ..., z_1N(1), ..., z_L1, ..., z_LN(L)
    // therefore, will be useful to consider the cumulative number of points to have a quick global access and still retrieve the number of points on the level with a simple difference.
    unsigned int * cumulative_points; cumulative_points = (unsigned int *) malloc((number_of_levels+1)*sizeof(unsigned int)); // cumulative points definition and allocation
    double * data_tree; // tree definition
    // compute cumulative number of points (this can be done explicitly for some values of mu, i.e. 0.5)
    cumulative_points[0] = 1;
    for (int level=0; level<number_of_levels; level++){cumulative_points[level+1]=cumulative_points[level]+pow(1+floor(pow(1/mu, level+1)), 2);}
    data_tree = (double *) malloc(cumulative_points[number_of_levels] * (POINTS_DIM+VALUES_DIM) * sizeof (double)); // tree allocation
    double * data_dev; CUDA_ERROR_CHECK(cudaMalloc(&data_dev, cumulative_points[number_of_levels] * (POINTS_DIM+VALUES_DIM) * sizeof (double))); // data allocation on device

    /// Parallel computation of the equispaced grid, level-wise
    // These variables are used to convert occupancy to warps
    int block_counter_grid_single; // number of blocks involved definition
    dim3 thread_per_block_grid(16,16,1); // since we deal with a 2d grid is convenient to introduce 2d threads
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&block_counter_grid_single, computeGridPoints2d, thread_per_block_grid.x*thread_per_block_grid.y, 0); // compute block occupancy
    block_counter_grid_single *= nSMs; // occupancy on the gpu
    int top_divisor = sqrt(block_counter_grid_single); // find the optimal size for 2d blocks, in order to maximize the usage,
    while (block_counter_grid_single%top_divisor){top_divisor--;} // e.i. (x, y) s.t. xy = block_counter
    dim3 block_counter_grid(top_divisor, block_counter_grid_single/top_divisor, 1); // definition and allocation of 2d block grid
    //printf("Occupancy: thread_per_block = %dx%d, block_counter_grid = %dx%d (%d)\n", thread_per_block_grid.x, thread_per_block_grid.y, block_counter_grid.x, block_counter_grid.y, block_counter_grid_single);
    // compute the equispaced grid on gpu
    clock_t start_time_grid_gen = clock(); // definition and assignment of cpu clock for generating points
    for (int level=0; level<number_of_levels; level++){ // compute the equispaced grid of points on level
        // compute equispaced points at level with a thread grid
        computeGridPoints2d<<<block_counter_grid, thread_per_block_grid>>>(&data_dev[cumulative_points[level]*(POINTS_DIM+VALUES_DIM)], pow(mu, level+1), 1+floor(pow(1/mu, level+1)), cumulative_points[level+1]-cumulative_points[level]);
    }

    printf("%d-grid generated, for a total of %d points, in %f sec\n", number_of_levels, cumulative_points[number_of_levels], (double) (clock()-start_time_grid_gen)/CLOCKS_PER_SEC);
    CUDA_ERROR_CHECK(cudaMemcpy(data_tree, data_dev, cumulative_points[number_of_levels] * (POINTS_DIM+VALUES_DIM) * sizeof (double), cudaMemcpyDeviceToHost)); //copy the computed data locally on the host
    CUDA_ERROR_CHECK(cudaFree(data_dev)); // we no longer need this memory on the device

    /// Parallel build of the kd tree on the points sets level-wise
    clock_t start_treeing_time = clock(); // definition and assignment of cpu clock for tree construction
    for (int level=0; level<number_of_levels;level++){ // build tree at level
        ParallelKDTreeStructure(&data_tree[cumulative_points[level]*(POINTS_DIM+VALUES_DIM)], cumulative_points[level+1]-cumulative_points[level], deviceId, 32);
    }
    printf("Time needed to build the tree on all levels: %f seconds\n", (double) (clock()-start_treeing_time)/CLOCKS_PER_SEC);

    /// Set up the storing path
    const char sourceFilePath[200] = __FILE__; // get the path to this file
    char * dirPath; // temporary string to store the current directory path
    char file_path_domain[200]; // path from the directory of this file to the directory where the results will be stored

    // Find the last occurrence of the directory separator character ("/" or "\")
    const char* lastSeparator = strrchr(sourceFilePath, '/'); // For Unix-like systems
    if (lastSeparator) {
        // Calculate the length of the directory path
        size_t dirPathLength = lastSeparator - sourceFilePath; // find the length of the path to the directory of this file
        dirPath = (char *) malloc((dirPathLength+40)*sizeof(char)); // allocate memory for the directory path
        strncpy(dirPath, sourceFilePath, dirPathLength); // copy the path till the directory
        dirPath[dirPathLength] = '\0'; // add end string
        size_t filePathLength = snprintf(file_path_domain, 200, "/%dD_domain/multiscale_times_L%d.txt", POINTS_DIM, number_of_levels); // define the path to the storing directory
        strcat(dirPath, file_path_domain); // append the path to the storing directory to the path to the file directory
        strncpy(file_path_domain, dirPath, dirPathLength+filePathLength); // copy dirPathLength+10 characters from dirPath to file_path_domain
        // a good remark is that here we cannot directly use dirPath because is defined inside the if-else, we need to reference to an external pointer, in this case file_path_domain.
        file_path_domain[dirPathLength+filePathLength] = '\0'; //enforce the string termination
        free(dirPath);
        // Print the resulting directory path
        printf("Storing file in: %s\n", file_path_domain);
    } else {
        printf("Failed to determine directory path: %s.\n", sourceFilePath);
        return EXIT_FAILURE;
    }

    // PARAMETERS ========================================================================================================================
    int wendland_coefficients[2] = {1,3}; // definition and assignment of wendland coefficients
    double * solution_vector = (double *) calloc(cumulative_points[number_of_levels], sizeof(double)); // solution definition and allocation
    // compute the approximation and store the times
    FILE *times_file_pointer; // pointer to the file
    times_file_pointer = fopen(file_path_domain, "w");
    // In our GPU we can have up to 32 warps per SM (compute capability 8.0) we chose to start from nSMs to do not have unnecessary long waiting time
    // Is worth notice that the code is efficient from given_warps = nSMs, before we leave many processors idling
    for (int given_warps = nSMs/4; given_warps < 32*nSMs+1; given_warps *= 2){
        clock_t sol_clock = clock(); // definition and assignment of cpu clock for system solution
        multiscale_parallel_interpolation(data_tree, cumulative_points, number_of_levels, mu, nu, EPS, wendland_coefficients, solution_vector, given_warps, nSMs);
        double sol_time = (double) (clock()-sol_clock)/CLOCKS_PER_SEC;
	printf("Solved the full system with %d warps in %f seconds \n", given_warps, sol_time);
        fprintf(times_file_pointer, "Solved the full system with %d warps in %f seconds \n", given_warps, sol_time);
    }
    fclose(times_file_pointer);
    free(solution_vector);
    free(data_tree);
    return EXIT_SUCCESS;
}
