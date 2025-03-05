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
    /// Expecting parameters are (ordered): Number of levels, warps per SM, mu and nu.
    // arguments definition
    unsigned int number_of_levels;
    double mu;
    double nu;
    unsigned int warps_per_SM;
    // arguments handling and allocation
    if (argc > 5 || argc == 1){
        fprintf(stderr, "Expecting 1 to 4 parameters, had %d.\n", argc-1);
        return EXIT_FAILURE;
    } else if (argc == 2){
        number_of_levels = atoi(argv[1]);
        warps_per_SM = 8;
        mu = 0.5;
        nu = 4;
    } else if (argc == 3){
        number_of_levels = atoi(argv[1]);
        warps_per_SM = atoi(argv[2]);
        mu = 0.5;
        nu = 4;
    } else if (argc == 4){
        number_of_levels = atoi(argv[1]);
        warps_per_SM = atoi(argv[2]);
        mu = atof(argv[3]);
        nu = 4;
    } else {
        number_of_levels = atoi(argv[1]);
        warps_per_SM = atoi(argv[2]);
        mu = atof(argv[3]);
        nu = atof(argv[4]);
    }

    /// device setup, gathering basic information
    int deviceId, nSMs;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&nSMs, cudaDevAttrMultiProcessorCount, deviceId);
    // cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, deviceId); we assumed that warp_size = 32.
    /// parameters of the gpu setup
    int given_warps = warps_per_SM*nSMs; // we want to tune the amount of resources that we will use but still exploit them consciously, therefore the number of warps will always be a multiple of the number of streaming multiprocessors.
    printf("Running with %d levels, %d given warps, mu = %f, nu = %f\n", number_of_levels, given_warps, mu, nu);

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
    char dir_path_domain[200]; // path from the directory of this file to the directory where the results will be stored

    // Find the last occurrence of the directory separator character ("/" or "\")
    const char* lastSeparator = strrchr(sourceFilePath, '/'); // For Unix-like systems
    if (lastSeparator) {
        // Calculate the length of the directory path
        size_t dirPathLength = lastSeparator - sourceFilePath; // find the length of the path to the directory of this file
        dirPath = (char *) malloc((dirPathLength+10)*sizeof(char)); // allocate memory for the directory path
        strncpy(dirPath, sourceFilePath, dirPathLength); // copy the path till the directory
        dirPath[dirPathLength] = '\0'; // add end string
        snprintf(dir_path_domain, 200, "/%dD_domain", POINTS_DIM); // define the path to the storing directory
        strcat(dirPath, dir_path_domain); // append the path to the storing directory to the path to the file directory
        strncpy(dir_path_domain, dirPath, dirPathLength+10); // copy dirPathLength+10 characters from dirPath to dir_path_domain
        // a good remark is that here we cannot directly use dirPath because is defined inside the if-else, we need to reference to an external pointer, in this case dir_path_domain.
        dir_path_domain[dirPathLength+10] = '\0'; //enforce the string termination
        free(dirPath);
        // Print the resulting directory path
        printf("Storing directory path: %s\n", dir_path_domain);
    } else {
        printf("Failed to determine directory path: %s.\n", sourceFilePath);
        return EXIT_FAILURE;
    }

    // PARAMETERS ========================================================================================================================
    int wendland_coefficients[2] = {1,3}; // definition and assignment of wendland coefficients
    double * solution_vector = (double *) calloc(cumulative_points[number_of_levels], sizeof(double)); // solution definition and allocation
    // compute the approximation and store it
    multiscale_parallel_interpolation(data_tree, cumulative_points, number_of_levels, mu, nu, EPS, wendland_coefficients, solution_vector, given_warps, nSMs);

    // s_j = sum_{i=1}^{N_j} alpha_i^(j) Phi_j(dot, \x_i^(j))
    /// store the approximant
    // set up domain for evaluation of the approximant
    double threshold = 1.0/(EVALUATION_POINTS_ON_AXIS-1); // the evaluation points are equispaced points on the domain with "threshold" as grid step.
    double temp_norm;
    // In the following we assume that our domain is within [0,1]^d.
    // We evaluate the approximation on an equispaced grid of EVALUATION_POINTS_ON_AXIS points along each axis for a total of EVALUATION_POINTS_ON_AXIS^POINTS_DIM points
    //printf("\nTime to test the accuracy of the approximation and save the interpolant for every level,\nThe files will be saved into ./%dD_domain/result_in_%d_points\n", POINTS_DIM, (int) pow(EVALUATION_POINTS_ON_AXIS, POINTS_DIM));

    double temp_value; // definition of temp_value for storing the approximant evaluation
    double delta = nu * mu/sqrt(2); // starting value for delta: delta_l = nu * h_l. Since we chose as data equispaced points on the unitary box with step mu^(level+1) for level=0...number_of_levels-1, we have that h_0 = mu / sqrt(2)
    clock_t save_time = clock(); // cpu clock for testing results
    for (int level = 0; level < number_of_levels; ++level) { // evaluate the approximant of "level"
        // create the file for data storage
        FILE *evaluated_file_pointer; // pointer to the file
        size_t length = strlen(dir_path_domain); // length of the directory path where data will be stored
        char * storage_path = (char*) malloc(length + 200 * sizeof(char)); // path to the file that will store the results, 200 characters should be enough to store our file name.
        strcpy(storage_path, dir_path_domain); // copy the path till the directory where the results will be stored
        char path[200]; // define a local path variable
        snprintf(path, 200, "/result_in_%d_points/approximation%d_%d_evaluated_in_domain_%dD.csv", (int) pow(EVALUATION_POINTS_ON_AXIS, POINTS_DIM), EVALUATION_POINTS_ON_AXIS, level+1, POINTS_DIM); // define the file name
        strcat(storage_path, path); // add the file name to the path of the directory
        // file created, now will be open with writing permission
        evaluated_file_pointer = fopen(storage_path, "w");

        double temp_grid_point[POINTS_DIM]; // definition per the point of our evaluation grid, to spare storage space we will go through them one by one
        for (int i = 0; i < POINTS_DIM; i++) { temp_grid_point[i] = 0; } // set upo initial values of the grid point, based in our "unitary box" settings
        for (int i = 0; i < pow(EVALUATION_POINTS_ON_AXIS, POINTS_DIM); i++) { // here we assumed still the unitary box settings, i.e. in every dimension there are the same number of evaluation points. Can be generalized.
            for (int d = 0; d < POINTS_DIM; d++) {
                fprintf(evaluated_file_pointer, "%f,", temp_grid_point[d]); // write on file the evaluation point entries
            }
            temp_value = 0; // initial value of the approximant on the grid point
            // evaluate the sum_{i=1}^{N_j} alpha_i^(j) Phi_j(domain_point, \x_i^(j)) for fixed level.
            for (int point_index = 0; point_index < cumulative_points[level+1]-cumulative_points[level]; point_index++) {
                // Set norm to zero then add for each dimension the squared difference
                temp_norm = 0;
                for (int temp_dim = 0; temp_dim<POINTS_DIM; temp_dim++){
                    temp_norm += pow(data_tree[cumulative_points[level]*(POINTS_DIM+VALUES_DIM)+point_index+temp_dim*(cumulative_points[level+1]-cumulative_points[level])]-temp_grid_point[temp_dim], 2);
                } // To have the norm now we need only to take the square root
                // Store values on the matrix in packed form (ord=RowMajor)
                temp_value += solution_vector[cumulative_points[level] + point_index] * wendland_function(sqrt(temp_norm) / delta, wendland_coefficients[0],wendland_coefficients[1]) / pow(delta, POINTS_DIM);
            }
            fprintf(evaluated_file_pointer, "%f\n", temp_value); // write on file the approximant result on grid point

            // increase the grid by one step
            temp_grid_point[0] += threshold;
            for (int d = 1; d < POINTS_DIM; d++) {
                if ((i + 1) % (int) pow(EVALUATION_POINTS_ON_AXIS, d) == 0) { // reached the end along axis "d"
                    temp_grid_point[d] += threshold;
                    if (d != 0) {
                        temp_grid_point[d - 1] = 0; // based in our "unitary box" settings, should be min along dim "d-1"
                    }
                }
            } // end increase
        }
        // results of level are stored on file, close the file and free resources
        fclose(evaluated_file_pointer);
        free(storage_path);
        clock_t save_time_delta = clock() - save_time;
        //printf(" saved at time: %f sec.\n", (double) save_time_delta/CLOCKS_PER_SEC);
        delta*=mu; // update delta
    }
    // all results are stored, free the solution

    free(solution_vector);
    free(data_tree);
    return EXIT_SUCCESS;
}
