#ifndef PARALLEL_MULTISCALE_APPROXIMATION_ITERATIVE_PARALLEL_METHODS_CUH
#define PARALLEL_MULTISCALE_APPROXIMATION_ITERATIVE_PARALLEL_METHODS_CUH

#include "macros.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cublas_v2.h" // https://docs.nvidia.com/cuda/cublas/index.html#
#include <cblas.h> // C. L. Lawson, R. J. Hanson, D. Kincaid, and F. T. Krogh, Basic Linear Algebra Subprograms for FORTRAN usage, ACM Trans. Math. Soft., 5 (1979), pp. 308—323. or // https://www.openblas.net/
#include "wendland_functions_device.cuh"
#include "multiscale_structure.cuh"
#define CUBLAS_ERROR_CHECK(err) do { cublas_check((err), __FILE__, __LINE__); } while(false)
inline void cublas_check(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUDA Error %d. In file '%s' on line %d\n", status, file, line);
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
}
#define MAX_DEPTH 32 // we assume to deal with trees with depth up to 32, i.e. up 2^32-1 elements

void writeToFileFromHost(const char * fileName, double * arrayPointer_host, int arrayLength, int arrayDim){
    //get the current path
    const char sourceFilePath[200] = __FILE__;
    char * dirPath;
    char dir_path_domain[40];

    // Find the last occurrence of the directory separator character ("/" or "\")
    const char* lastSeparator = strrchr(sourceFilePath, '/'); // For Unix-like systems
    if (lastSeparator) {
        // Calculate the length of the directory path
        size_t dirPathLength = lastSeparator - sourceFilePath;
        dirPath = (char *) malloc((dirPathLength+40)*sizeof(char));
        strncpy(dirPath, sourceFilePath, dirPathLength);
        dirPath[dirPathLength] = '\0';
        snprintf(dir_path_domain, 40, "/%s", fileName);
        strcat(dirPath, dir_path_domain);
        //dir_path_domain[dirPathLength+10] = '\0';
        // Print the resulting directory path
        printf("Storing directory path: %s\n", dirPath);
    } else {
        printf("Failed to determine directory path: %s.\n", sourceFilePath);
    }
    //save data on file
    FILE *evaluated_file_pointer;
    evaluated_file_pointer = fopen(dirPath, "w");
    for (int i=0; i != arrayLength; i++){
        fprintf(evaluated_file_pointer, "%d: ", i);
        for (int d=0; d != arrayDim; d++){
            fprintf(evaluated_file_pointer, "%8.6f,", arrayPointer_host[i+d*arrayLength]);
        }
        fprintf(evaluated_file_pointer, "\n");
    }
    fprintf(evaluated_file_pointer, "\n");
    fclose(evaluated_file_pointer);
    free(dirPath);
}

__device__ void stack_based_balanced_tree_contribution(double * zero_tree_pointer, unsigned int tree_depth, double * zero_reference_node_pointer, int reference_node_index, unsigned int number_of_points, unsigned int number_of_points_reference, double alpha, double * vector, double * result, double delta, double wendland_k, int wendland_d) {
    /// THis function is supposed to be called to compute the row vector multiplication, where both sets involved (row and column) can be associated with a kd tree of points.
    /// given a reference_node, defined by a pointer to its array and its index, and the pointer to a kd-tree, it will traverse the tree to update the result entry associated to the reference node: result[reference_node_index] += alpha*vector[node]*Phi_delta(node, reference_node), for ever node such that dist(node,reference_node) < delta.
    /// the traversal of the tree is made by keeping track of the points that i should visit.
    /// zero_tree_pointer: pointer to the tree to be traversed.
    /// tree_depth: depth of the tree to be traversed.
    /// zero_reference_node_pointer: pointer to the tree of the reference node.
    /// reference_node_index: index of the reference node in its tree.
    /// number_of_points: number of points on tree to be traversed
    /// number_of_points_reference: number of points on tree of the reference node
    /// alpha: coefficients involved in the update
    /// vector: vector involved in the matrix vector multiplication related to this function call.
    /// result: solution vector involved in the matrix vector multiplication. the value of the result vector with index "reference_node_index" will be updated as describe above.
    /// delta: radius of the support of the compactly supported rbf. phi_delta(x) = 0 if ||x||_2^2 > delta.
    /// wendland_coefficients. coefficients chosen of the wendland function we operate with.

    /// complexity notes:
    /// space on device: O(N): 1
    /// time: O(N): log2(N)
    int to_visit_list[MAX_DEPTH], to_length_list[MAX_DEPTH], to_axis_list[MAX_DEPTH];
    // to_axis_list can be replaced using the information on the length of the subtree: axis = (depth-log2(tree_length)-1)%POINTS_DIM
    double squared_norm = 0; double diff_axis;
    // test the root
    for (int d=0; d<POINTS_DIM; d++) { squared_norm += pow(zero_tree_pointer[number_of_points/2+d*number_of_points] - zero_reference_node_pointer[reference_node_index+d*number_of_points_reference], 2); }
    if (squared_norm < delta*delta) { // update the result
        result[reference_node_index] += alpha*pow(delta,-POINTS_DIM)*vector[number_of_points/2]*wendland_function_device(sqrt(squared_norm)/delta, wendland_k, wendland_d);
    }
    // then add to the waiting list the children of the root since a priori no one can be excluded
    to_visit_list[1] = number_of_points/2-(number_of_points+2)/4; to_length_list[1] = number_of_points/2; to_axis_list[1] = 1;
    to_visit_list[0] = number_of_points/2+(number_of_points+3)/4; to_length_list[0] = (number_of_points-1)/2; to_axis_list[0] = 1;
    int visit_index = 1;
    while (visit_index>=0){
        squared_norm = 0;
        for (int d=0; d<POINTS_DIM; d++) { squared_norm += pow(zero_tree_pointer[to_visit_list[visit_index]+d*number_of_points] - zero_reference_node_pointer[reference_node_index+d*number_of_points_reference], 2); }
        if (squared_norm < delta*delta) { // update the result
            result[reference_node_index] += alpha*pow(delta,-POINTS_DIM)*vector[to_visit_list[visit_index]]*wendland_function_device(sqrt(squared_norm)/delta, wendland_k, wendland_d);
        }
        if (to_length_list[visit_index] == 1) {// leaf case
            visit_index--;
        } else if (to_length_list[visit_index] == 2) {//node with one child (the left one)
            to_visit_list[visit_index] -= (to_length_list[visit_index]+2)/4;
            to_length_list[visit_index] /= 2;
            to_axis_list[visit_index] = (to_axis_list[visit_index]+1)%POINTS_DIM;
        } else { // inner node
            diff_axis = zero_tree_pointer[to_visit_list[visit_index]+to_axis_list[visit_index]*number_of_points]-zero_reference_node_pointer[reference_node_index+to_axis_list[visit_index]*number_of_points_reference];
            if (diff_axis > 0) {// the left child is closer (actually closer is not really correct, more likely to give useful results suits better)
                // firstly check if also the right child should be explored
                // indeed, adding the left child  on the current point will alter the values needed to the other child determination
                // on the other hand, if the right child is not meaningful to visit we have to store the left one over the current point
                if (diff_axis * diff_axis <= delta * delta){
                    // yes, we add the right child to the visit list
                    to_visit_list[visit_index+1] = to_visit_list[visit_index]+(to_length_list[visit_index]+3)/4;
                    to_length_list[visit_index+1] = (to_length_list[visit_index]-1)/2;
                    to_axis_list[visit_index+1] = (to_axis_list[visit_index]+1)%POINTS_DIM;
                    // and of course the left child
                    to_visit_list[visit_index] -= (to_length_list[visit_index]+2)/4;
                    to_length_list[visit_index] = to_length_list[visit_index]/2;
                    to_axis_list[visit_index] = to_axis_list[visit_index+1];
                    visit_index++;
                } else { // otherwise, we add just the left child to the visit list
                    to_visit_list[visit_index] -= (to_length_list[visit_index]+2)/4;
                    to_length_list[visit_index] = to_length_list[visit_index]/2;
                    to_axis_list[visit_index] = (to_axis_list[visit_index]+1)%POINTS_DIM;
                } // end if-else which-child-explore
            } else { // the right child is the closer
                // similarly to the other case, firstly check if also the left child should be explored
                if (diff_axis * diff_axis <= delta * delta) {
                    // yes, we add the left child to the visit list
                    to_visit_list[visit_index+1] = to_visit_list[visit_index]-(to_length_list[visit_index]+2)/4;
                    to_length_list[visit_index+1] = to_length_list[visit_index]/2;
                    to_axis_list[visit_index+1] = (to_axis_list[visit_index]+1)%POINTS_DIM;
                    // and of course the right child
                    to_visit_list[visit_index] += (to_length_list[visit_index]+3)/4;
                    to_length_list[visit_index] = (to_length_list[visit_index]-1)/2;
                    to_axis_list[visit_index] = to_axis_list[visit_index+1];
                    visit_index++;
                } else{ // otherwise, we add just the right child to the visit list
                    to_visit_list[visit_index] += (to_length_list[visit_index]+3)/4;
                    to_length_list[visit_index] = (to_length_list[visit_index]-1)/2;
                    to_axis_list[visit_index] = (to_axis_list[visit_index]+1)%POINTS_DIM;
                } // end if-else which-child-explore
            } // end if-else closer-to-reference-child
        } //end if-elif-else leaf-double-inner node
    } // end while there are still points to visit
}

__global__ void partial_mv_row_point_based(double * row_set, int row_points, double * column_set, int column_points, unsigned int column_tree_depth, double alpha, double * v, double * result, double delta, double wendland_k, int wendland_d) {
    // every thread will compute a single row @ vector exploiting the kd structure
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < row_points; k += gridDim.x * blockDim.x) {
        stack_based_balanced_tree_contribution(column_set, column_tree_depth, row_set, k, column_points, row_points, alpha, v, result, delta, wendland_k, wendland_d);
    }
}

__global__ void partial_B_mv_row_point_based(double * row_full_set, unsigned int * cumulative_points, unsigned int number_of_levels, double * column_set, unsigned int column_level, unsigned int column_tree_depth, double alpha, double * v, double * result, double delta, double wendland_k, int wendland_d) {
    // every thread will compute a single row @ vector exploiting the kd structure. However, since it's meaningful to know on which level the point is (to exploit the associated kd tree) we also invest some operation to find this information.
    int level = column_level+1;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < cumulative_points[number_of_levels]-cumulative_points[column_level+1]; k += gridDim.x * blockDim.x) {
        // update the level associated with the point we are currently on
        while (k >= cumulative_points[level+1]-cumulative_points[column_level+1]){level++;}
        stack_based_balanced_tree_contribution(column_set, column_tree_depth, &row_full_set[cumulative_points[level]-cumulative_points[column_level+1]], k, cumulative_points[column_level+1]-cumulative_points[column_level], cumulative_points[level+1]-cumulative_points[level], alpha, v, result, delta, wendland_k, wendland_d);
    }
}

void B_column_based_mv(double * points, unsigned int * cumulative_size, unsigned int number_of_levels, unsigned int column_level, double alpha, double * vector, double * result, double mu, double nu, double wendland_k, double wendland_d, unsigned int warps_per_block, unsigned int blocks){
    /// compute solution = solution + alpha * Matrix @ vector.
    /// points. the full set of points properly stored.
    /// cumulative_size. the list of the cumulative sizes of the matrix with respect to the column_level.
    /// column_level. level of the column i.e. of the tree.
    /// alpha. coefficient multiplied.
    /// vector. vector multiplied by the matrix. full vector.
    /// solution. solution updated by the matrix vector multiplication. full vector.
    /// wendland_coefficients. coefficients chosen of the wendland function we operate with.
    /// mu: coefficient of uniform decrease of the fill distance of the level sets: c_h h_l mu <= h_{l+1} <= mu h_l
    /// nu: coefficient of proportionality between the support of the compactly supported rbf and the fill distance of the set: delta_l = nu h_l
    /// warps_per_block. available warps per block for the computations.
    /// blocks. available blocks for the computations.
    /// nSMs. number of streaming multiprocessors available on the gpu.

    /// complexity notes: (N = total length, n = column length)
    /// space on device: O(N): (d+1)N doubles + L ints.
    /// time: O(N): nlog2(N-n)

    // definition of pointers to device data
    double * row_device, * column_set_device; double * vector_device, * result_device;
    // allocation of memory on the device
    CUDA_ERROR_CHECK(cudaMalloc(&column_set_device, (cumulative_size[column_level+1]-cumulative_size[column_level])*(POINTS_DIM)*sizeof(double))); // points from column_level
    CUDA_ERROR_CHECK(cudaMalloc(&row_device, (cumulative_size[number_of_levels]-cumulative_size[column_level+1])*(POINTS_DIM)*sizeof(double))); // all points from levels greater that column_level (excluded)
    CUDA_ERROR_CHECK(cudaMalloc(&vector_device, (cumulative_size[column_level+1]-cumulative_size[column_level])*VALUES_DIM*sizeof(double))); // vector to be multiplied with the column-block-matrix
    CUDA_ERROR_CHECK(cudaMalloc(&result_device, (cumulative_size[number_of_levels]-cumulative_size[column_level+1])*(VALUES_DIM)*sizeof(double))); // result vector
    // copy points of the column_level set
    CUDA_ERROR_CHECK(cudaMemcpy(column_set_device, &points[cumulative_size[column_level]*(POINTS_DIM+VALUES_DIM)], (cumulative_size[column_level+1]-cumulative_size[column_level])*(POINTS_DIM)*sizeof(double), cudaMemcpyHostToDevice));
    // copy of the vector to be multiplied with the matrix
    CUDA_ERROR_CHECK(cudaMemcpy(vector_device, &vector[cumulative_size[column_level]*(VALUES_DIM)], (cumulative_size[column_level+1]-cumulative_size[column_level])*(VALUES_DIM)*sizeof(double), cudaMemcpyHostToDevice));
    // copy of the solution that will be updated adding alpha * matrix @ vector.
    CUDA_ERROR_CHECK(cudaMemcpy(result_device, &result[cumulative_size[column_level+1]*(VALUES_DIM)], (cumulative_size[number_of_levels]-cumulative_size[column_level+1])*(VALUES_DIM)*sizeof(double), cudaMemcpyHostToDevice));
    // copy of the points from levels greater than column_level
    for (int level=column_level+1; level<number_of_levels; level++){
        CUDA_ERROR_CHECK(cudaMemcpy(&row_device[(cumulative_size[level]-cumulative_size[column_level+1])*(POINTS_DIM)], &points[cumulative_size[level]*(POINTS_DIM+VALUES_DIM)], (cumulative_size[level+1]-cumulative_size[level])*(POINTS_DIM)*sizeof(double), cudaMemcpyHostToDevice));
    }
    // definition, allocation and copy of the cumulative_size array on device
    unsigned int * cumulative_size_device; CUDA_ERROR_CHECK(cudaMalloc(&cumulative_size_device, (number_of_levels+1)*sizeof(int)));
    CUDA_ERROR_CHECK(cudaMemcpy(cumulative_size_device, cumulative_size, (number_of_levels+1)*sizeof(int), cudaMemcpyHostToDevice));
    /// computation step
    partial_B_mv_row_point_based<<<blocks, 32*warps_per_block>>>(row_device, cumulative_size_device, number_of_levels, column_set_device, column_level, log2(cumulative_size[column_level+1]-cumulative_size[column_level])+1, alpha, vector_device, result_device, nu * mu/sqrt(2) * pow(mu,column_level), wendland_k, wendland_d);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // synchronize the threads
    // copy the result back on the host
    CUDA_ERROR_CHECK(cudaMemcpy(&result[cumulative_size[column_level+1]*(VALUES_DIM)], result_device, (cumulative_size[number_of_levels]-cumulative_size[column_level+1])*(VALUES_DIM)*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(row_device));
    CUDA_ERROR_CHECK(cudaFree(column_set_device));
    CUDA_ERROR_CHECK(cudaFree(vector_device));
    CUDA_ERROR_CHECK(cudaFree(result_device));
    CUDA_ERROR_CHECK(cudaFree(cumulative_size_device));
};

void concurrent_cgA(double * points, unsigned int * cumulative_size, double * rhs, double eps, double * cg_x, unsigned int * level_list, unsigned int * warps_per_block_list, unsigned int * block_list, unsigned int nSMs, unsigned int list_length, cudaStream_t * streams_list, double mu, double nu, double wendland_k, int wendland_d) {
    /// compute the conjugate gradient of the system A_l x_l = b_l, for all levels "l" in level_list.
    /// points. the full set of points properly stored.
    /// cumulative_size. the list of the cumulative sizes of the matrix with respect to the level.
    /// rhs. full rhs store conveniently as the points.
    /// eps. related to the cg threshold.
    /// cg_x. initial solution for the cg and where the final solution will be stored. full vector.
    /// level_list. array of the levels involved in this concurrent cg.
    /// warps_per_block_list. array of the warps per block needed, with respect to the level of level_list, in order solve efficiently the cg.
    /// block_list. array of the blocks employed, with respect to the level of level_list, in order solve efficiently the cg.
    /// nSMs. number of streaming multiprocessors available on the gpu.
    /// list_length. number of levels involved.
    /// streams_list. array of streams in order to compute concurrently operation related to cg steps of different levels. Every kernel should be launched in a different stream but not into a specific one, is just to run the kernels simultaneously.
    /// wendland_coefficients. coefficients chosen of the wendland function we operate with.
    /// mu: coefficient of uniform decrease of the fill distance of the level sets: c_h h_l mu <= h_{l+1} <= mu h_l
    /// nu: coefficient of proportionality between the support of the compactly supported rbf and the fill distance of the set: delta_l = nu h_l
    // from Numerical Linear Algebra, An Introduction, by Holger Wendland (2017) Algorithm 22 (p.195).

    /// complexity notes:   (N = sum length(level) for level in list)
    /// space on device: O(N): (d+4)N doubles.
    /// time: O(N): kNlog2(N)
    // CuBLAS handle definition and creation.
    cublasHandle_t handle; CUBLAS_ERROR_CHECK(cublasCreate(&handle));

    // definition and allocation of variables for the solution. For each constant there is an array of list_length elements
    double * beta, * alpha, * m_alpha, * residual_norm;
    alpha = (double *) malloc(list_length*sizeof(double));
    m_alpha = (double *) malloc(list_length*sizeof(double));
    beta = (double *) malloc(list_length*sizeof(double));
    residual_norm = (double *) malloc(list_length*sizeof(double));
    const double one = 1.0; // one is defined as variable because cublas requires that every parameter is passed by its address.
    // double pointer for all the arrays required in the cg solution, each element points to the pointer of the array of the associated level.
    double ** residual_dev, ** descent_dir_dev, ** temp_vector_dev, ** points_dev, ** result_dev;
    residual_dev = (double **) malloc(list_length*sizeof(double *));
    descent_dir_dev = (double **) malloc(list_length*sizeof(double *));
    result_dev = (double **) malloc(list_length*sizeof(double *));
    temp_vector_dev = (double **) malloc(list_length*sizeof(double *));
    points_dev = (double **) malloc(list_length*sizeof(double *));
    // We know from theoretical results that the number of cg iteration is bounded independently of the level. However, we can still have a different amount of iteration for different levels.
    short int * convergence_index; // definition of pointer to an array of booleans to keep track of which level converged
    convergence_index = (short int *) malloc(list_length*sizeof(short int));
    short int sum_convergence_index = list_length; // definition of variable to check how many levels are solved.

    // resource allocation on the device
    for (int i=0;i<list_length;i++){
        // for every level allocate a proper amount of resources on the device
        convergence_index[i] = 1; // not converged yet
        CUDA_ERROR_CHECK(cudaMalloc(&points_dev[i], (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]])*(POINTS_DIM)*sizeof(double))); // points on device
        CUDA_ERROR_CHECK(cudaMalloc(&residual_dev[i], (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]])*sizeof(double))); // residual vector on device
        CUDA_ERROR_CHECK(cudaMalloc(&descent_dir_dev[i], (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]])*sizeof(double))); // direction vector on device
        CUDA_ERROR_CHECK(cudaMalloc(&temp_vector_dev[i], (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]])*sizeof(double))); // temporary vector on device
        CUDA_ERROR_CHECK(cudaMalloc(&result_dev[i], (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]])*sizeof(double))); // solution vector on device

        // create a cublas stream for the level
        CUBLAS_ERROR_CHECK(cublasSetStream(handle, streams_list[i]));
        // Copy points on device, then rhs to residual and descent_dir.
        CUDA_ERROR_CHECK(cudaMemcpyAsync(points_dev[i], &points[cumulative_size[level_list[i]]*(POINTS_DIM+VALUES_DIM)], (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]])*(POINTS_DIM)*sizeof(double), cudaMemcpyHostToDevice, streams_list[i]));
        CUDA_ERROR_CHECK(cudaMemcpyAsync(residual_dev[i], &rhs[cumulative_size[level_list[i]]*VALUES_DIM], (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]])*sizeof(double), cudaMemcpyHostToDevice, streams_list[i]));
        CUDA_ERROR_CHECK(cudaMemcpyAsync(descent_dir_dev[i], &rhs[cumulative_size[level_list[i]]*VALUES_DIM], (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]])*sizeof(double), cudaMemcpyHostToDevice, streams_list[i]));
        // Lastly copy the initial solution vector and set the first residual_norm (<res_vector, res_vector>).
        CUDA_ERROR_CHECK(cudaMemcpyAsync(result_dev[i], &cg_x[cumulative_size[level_list[i]]*VALUES_DIM], (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]])*sizeof(double), cudaMemcpyHostToDevice, streams_list[i]));
        CUBLAS_ERROR_CHECK(cublasDdot(handle, cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]], residual_dev[i], 1, residual_dev[i], 1, &residual_norm[i])); // set the first residual
        // Remark: we use the squared norm even the variable is called simply residual norm, since in many steps is exactly what is required.
	}
    /// cg iteration notes
    // every step in the routine is proceeded by this piece of code:
    // for (int i=0;i<list_length;i++){
    //            if (convergence_index[i]){
    // indeed, every step is concurrently computed between the chosen levels (for loop) that are not converged yet (convergence_index = 1)
    // the synchronization blocks (cudaDeviceSynchronize) guarantee that every stream has done the previous step before going further.
    do {
        // Initialize the temp_vector to zero
        for (int i=0;i<list_length;i++){
            if (convergence_index[i]){
                CUDA_ERROR_CHECK(cudaMemsetAsync(temp_vector_dev[i], 0, (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]]) * sizeof(double), streams_list[i]));
                // Perform matrix-vector multiplication kd-tree optimized function
                partial_mv_row_point_based<<<block_list[i], 32*warps_per_block_list[i], 0, streams_list[i]>>>(points_dev[i], (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]]), points_dev[i], (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]]), log2((cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]]))+1, 1, descent_dir_dev[i], temp_vector_dev[i], nu * mu/sqrt(2) * pow(mu,level_list[i]),  wendland_k, wendland_d);
            }
        }
        for (int i=0;i<list_length;i++){
            if (convergence_index[i]) {
                // Compute <t, p>, to then compute alpha
                CUBLAS_ERROR_CHECK(cublasSetStream(handle, streams_list[i]));
                CUBLAS_ERROR_CHECK(cublasDdot(handle, (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]]), temp_vector_dev[i], 1, descent_dir_dev[i], 1, &alpha[i]));
            }
        }
        CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Synchronize the threads
        for (int i=0;i<list_length;i++){
            if (convergence_index[i]) {
                CUBLAS_ERROR_CHECK(cublasSetStream(handle, streams_list[i]));
                // Compute alpha = residual_norm / <t, p>
                alpha[i] = residual_norm[i] / alpha[i];
                m_alpha[i] = -alpha[i]; // since we have to pass the address of the value and not the value, we have to define a variable for "minus alpha"
                // Update cg_x: cg_x = cg_x + ap
                CUBLAS_ERROR_CHECK(cublasDaxpy(handle, (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]]), &alpha[i], descent_dir_dev[i], 1, result_dev[i], 1));
                // Update residual: r = r - at
                CUBLAS_ERROR_CHECK(cublasDaxpy(handle, (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]]), &m_alpha[i], temp_vector_dev[i], 1, residual_dev[i], 1));
                // Compute ||r||, to then compute beta
                CUBLAS_ERROR_CHECK(cublasDdot(handle, (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]]), residual_dev[i], 1, residual_dev[i], 1, &beta[i]));
            }
        }
        CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Synchronize the threads
        for (int i=0;i<list_length;i++){
            if (convergence_index[i]) {
                CUBLAS_ERROR_CHECK(cublasSetStream(handle, streams_list[i]));
                // Compute beta = ||r|| / residual_norm
                beta[i] = beta[i] / residual_norm[i];
                // Update descent_dir: p = bp + r
                CUBLAS_ERROR_CHECK(cublasDscal(handle, (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]]), &beta[i], descent_dir_dev[i], 1));
                CUBLAS_ERROR_CHECK(cublasDaxpy(handle, (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]]), &one, residual_dev[i], 1, descent_dir_dev[i], 1));
                CUDA_ERROR_CHECK(cudaMemcpyAsync(descent_dir_dev[i], descent_dir_dev[i], (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]]) * sizeof(double), cudaMemcpyHostToDevice, streams_list[i]));
                // Compute residual_norm = ||r||_2^2
                CUBLAS_ERROR_CHECK(cublasDdot(handle, (cumulative_size[level_list[i]+1]-cumulative_size[level_list[i]]), residual_dev[i], 1, residual_dev[i], 1, &residual_norm[i]));
            }
        }
        CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Synchronize the threads
        for (int i=0;i<list_length;i++){
            // check convergence and eventually change to zero the associated convergence_index
            if (convergence_index[i] && sqrt(residual_norm[i]) < eps){ // residual_norm is actually the squared norm (||r||_2^2) therefore we take into account sqrt(residual_norm)
                convergence_index[i]=0;
		        sum_convergence_index--;
            }
        }
    } while (sum_convergence_index); // until convergence_sum =! 0, i.e. all levels are converged
    // copy the result on the cpu
    for (int i=0;i<list_length;i++) {
        CUDA_ERROR_CHECK(cudaMemcpyAsync(&cg_x[cumulative_size[level_list[i]] * VALUES_DIM], result_dev[i], (cumulative_size[level_list[i] + 1] - cumulative_size[level_list[i]])*sizeof(double), cudaMemcpyDeviceToHost, streams_list[i]));
    }
    CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Synchronize the threads
    // Destroy the cublas helper
    CUBLAS_ERROR_CHECK(cublasDestroy(handle));
    // Free memory within the device
    for (int i=0;i<list_length;i++){
        CUDA_ERROR_CHECK(cudaFree(points_dev[i]));
        CUDA_ERROR_CHECK(cudaFree(residual_dev[i]));
        CUDA_ERROR_CHECK(cudaFree(descent_dir_dev[i]));
        CUDA_ERROR_CHECK(cudaFree(temp_vector_dev[i]));
        CUDA_ERROR_CHECK(cudaFree(result_dev[i]));
    }
    // lastly free pointers to cpu memory
    free(beta); free(alpha); free(residual_norm); free(convergence_index);
    free(points_dev); free(residual_dev); free(descent_dir_dev); free(temp_vector_dev); free(result_dev);
}

void multiscale_parallel_interpolation(double * sequence_dataset, unsigned int * cumulative_points, unsigned int number_of_levels, double mu, double nu, double eps, int wendland_coefficients[2], double * solution, unsigned int nWarps, unsigned int nSMs) {
    /// sequence_dataset: pointer to the sequence of number_of_levels point sets stored as outlined in "data allocation" on main. We expect the data stored in the first structure and then we will use the second to speed up the parallel access to them in the system solution.
    /// cumulative_points: pointer to the cumulative values of the point sets, i.e. 0, N(0), N(0)+N(1), N(0)+N(1)+N(2), etc.
    /// number_of_levels: number of levels for the multiscale approximations
    /// mu: coefficient of uniformity decrease for the mesh norm, i.e. c_h h_j mu <= h_{j+1} <= h_j mu, where h_j is the mesh norm at the level j and c_h is a constant in (0,1].
    /// /// nu: The coefficient that define the radius (delta) of the compactly supported rbf, delta_j = nu * h_j, where h_j is the mesh norm at the level j.
    /// eps: convergence threshold;
    /// wendland_coefficients: A couple of integers (n, k) associated with the wanted Wendland compactly supported radial basis function phi_(n, k).
    /// solution: we expect store the to-be-solution here (of size cumulative_points[number_of_levels]). It will be exploited to store some temporary solutions. It initial value will be take as starting value for the first cg
    /// nWarps: number of warps exploited to solve the approximation
    /// nSMs: number of streaming multiprocessors on the gpu
    short int level; // definition of variable for iterating over levels
    /// Allocation of required resources and setup
    double * rhs, * beta_i; // definition of vector needed during the system solution
    // in rhs will store the multi-level rhs of the system
    rhs = (double *) malloc(cumulative_points[number_of_levels]*VALUES_DIM*sizeof(double));
    // vector beta_i will be employed first in the jacobi solution and then in the block diagonal system
    beta_i = (double *) malloc(cumulative_points[number_of_levels]*VALUES_DIM*sizeof(double)); // allocation
    // set up the streams to allow concurrent operations and the flow_index to keep track of which level is already computed and which one has to be computed
    unsigned short int * flow_index = (unsigned short int *) malloc(number_of_levels*sizeof(unsigned short int));
    cudaStream_t * streams = (cudaStream_t *) malloc(number_of_levels*sizeof(cudaStream_t));
    for (level = 0; level < number_of_levels; level++) {
        // store the values on the rhs
        memcpy(&rhs[cumulative_points[level]*VALUES_DIM], &sequence_dataset[cumulative_points[level]*VALUES_DIM+cumulative_points[level+1]*POINTS_DIM], (cumulative_points[level+1]-cumulative_points[level])*VALUES_DIM*sizeof(double));
        // and create the streams
        CUDA_ERROR_CHECK(cudaStreamCreate(&streams[level]));
    }
    short int inner_level; // comparing variable to find which level can be executed concurrently with the one scheduled
    short int level_counter = 0; // to keep track of how many levels we solve concurrently
    unsigned int available_warps; // to keep track of how many warps can still be exploited
    unsigned int time_scheduled_for_main; // estimate solution time of the main level with nWarps
    unsigned int time_subscheduled; // estimated solution time at comparison level with available_warps
    // once we estimated the number of warps needed for the comparison level we split them between used_warps_per_block and used_blocks
    unsigned int used_warps_per_block; // number of warps exploited block-wise
    unsigned int used_blocks; // number of block needed
    double needed_warps; // variable needed to find out how to distribute properly the warps
    // variable needed to store the info on the findings and use the in the function "distributed_cg"
    unsigned int * warps_per_block_needed_at_level = (unsigned int *) malloc(number_of_levels*sizeof(unsigned int));
    unsigned int * blocks_needed_at_level = (unsigned int *) malloc(number_of_levels*sizeof(unsigned int));
    unsigned int * levels_to_do = (unsigned int *) malloc(number_of_levels*sizeof(unsigned int));

    /// Jacobi solution strategy
    // we want to solve the system T b = f.
    // the Jacobi iteration is b_ip1 = f+(id-T)b_i. Moreover, we skip the first step where the null vector is chosen as starting vector and directly take b_0 = f (aka rhs).
    // we can compute (id-T)b_i = - B D(A^-1) b_i solving first the cg D(A) t_i = b_i. then multiplying the results with the B matrix and subtract it to f lead the desired result:
    // f - B t_i. For this reason we design our matrix vector multiplication in such way that does not simply compute the operation but directly add the result multiplied by a constant to a given vector.
    // Since the cg result is just a temporary result needed for the matrix-vector multiplication we store its result in the solution vector, then at the start of every iteration is set back to zeros.

    // copy the rhs them on beta_i, since beta_0 = rhs in the jacobi routine
    cblas_dcopy(cumulative_points[number_of_levels]*VALUES_DIM, rhs, 1, beta_i, 1);
    //set to 0 the flow_indexes (all levels still to be computed)
    memset(flow_index, 0, number_of_levels * sizeof(unsigned short int));

    /// distributed cg routine, step 1: find out how to distribute the warps.
    // the idea of the following piece of code is the following:
    // we assume the time needed to solve a cg on level is proportional to number_of_points[level], and inversely proportional to the processors employed.
    // therefore we start from the largest (useful) level, and compute time_scheduled_for_main, which gives a rough idea about how many loops the processors had to do to at every iteration of the cg.
    // given this number we estimate how warps are needed from the ones available in such way that their workload can be effectively distributed between the blocks.
    // after having store all this information we go further and try to see if smaller set can be computed concurrently with the largest set.
    // to figure this out we go through the same process, estimating on smaller levels their "required computational time" with available warps, and check whether it requires less or equal time.
    // with a positive answer we compute the warps/block distribution as before and go to even smaller levels, until either there are no available warps or we checked all levels.
    // now we solve the cgs concurrently using the streams previously created. Then, we start again the whole process, from the largest level that need still needs to be computed, with the aid of flow_index that keeps track of whether a level is solved or not.
    // the process is repeated until we solved the systems over all levels.
    level = number_of_levels-2; // the result of the cg of the last block is meaningless since it will go against a 0 matrix in the step of the matrix vector multiplication.
    do { // in the following 32 is the warp_size!
        // set initial resources, the higher the level the higher the priority (to larger sets to smaller), so given a level and estimated the resources and the time needed we try to concurrently solve smaller systems
        available_warps=nWarps; // initial warps resources
        inner_level = level; // highest unsolved level
        // estimate the "computational time" of level with nWarps
        time_scheduled_for_main = (int) ceil((cumulative_points[inner_level+1]-cumulative_points[inner_level])/(32.0*available_warps)); // here (an in the following estimations) we assume that the solution time is linear over the number of points on level
        // is worth to remark that even if the computational time of our matrix-vector multiplication is NlogN (and not linear over N), logN will still be small (up to ~32) therefore we expect this to still be a convenient and not impairing simplification, moreover, the other operation involved are linear in complexity over N.
        if (!flow_index[level]){ // check whether the level is still to be solved
            do {
                // estimate the "computational time" of inner_level with available_warps
                time_subscheduled = (int) ceil((cumulative_points[inner_level+1]-cumulative_points[inner_level])/(32.0*available_warps));
                // check whether the time estimation is smaller that the highest level estimate and the inner_level is still to be solved
                if (time_subscheduled <= time_scheduled_for_main && !flow_index[inner_level]) {
                    // add to the to do list of levels to compute concurrently the inner_level
                    levels_to_do[level_counter]=inner_level;
                    // update the available warps, not based on what is needed but on what we actually use (this quantities may differ since warps are effectively used only when the workload is correctly distributed on the SMs).
                    needed_warps = ceil((cumulative_points[inner_level+1]-cumulative_points[inner_level])/(32.0*time_scheduled_for_main));
                    // compute the number of warps per block needed. Since the aim is a clever resources distribution we prioritize the distribution over the blocks,
                    // i.e. with the need of 64 warps we have 64 blocks with 1 warp rather than 1 block with 64 warps
                    used_warps_per_block = (int) ceil(needed_warps/(nSMs*64));
                    // given the number of used_warps_per_block we compute the number of blocks that we need
                    used_blocks = (int) ceil(needed_warps/used_warps_per_block);
                    // remove from the available_warps the warps that we will deploy for this level
                    available_warps -= used_blocks*used_warps_per_block;
                    // add the needed warps and blocks for this level to the list:
                    warps_per_block_needed_at_level[level_counter]=used_warps_per_block;
                    blocks_needed_at_level[level_counter]=used_blocks;
                    // update the index to notify that this level should not be considered anymore
                    flow_index[inner_level]=1;
                    // increase the level_counter and reduce the inner_level to look if other levels can still being executed concurrently
                    level_counter++;
                    inner_level--;
                } else {inner_level--;} // if the level either do not fit (its expected time is high) or is already solved go further down.
                // continue the inner_level loop until either we run out of available_warps or all levels were investigated
            } while (available_warps > 0 && inner_level >=0); // end do while
            /// distributed cg routine, step 2: compute the chosen levels.
            // solve A_level (t)_level = (b_0)_level = f_level for the chosen levels. We store the solution the future solution vector.
            concurrent_cgA(sequence_dataset, cumulative_points, rhs, eps, solution, levels_to_do, warps_per_block_needed_at_level, blocks_needed_at_level, nSMs, level_counter, streams, mu, nu, wendland_coefficients[0], wendland_coefficients[1]);
        } // end if (we dealt with the higher unsolved level)
        level--; // decrease the highest unsolved level
        level_counter=0; // set to 0 the number of levels to be executed concurrently
    } while (level >= 0); // end distributed cg routine (we went through until there isn't an unsolved level)

    /// distributed matrix vector multiplication routine: for every block-column compute the matrix vector multiplication and update the result.
    // Given the structure and the number of points it will be always more efficient to deal compute it in this way as long as there is enough space on the gpu for the data.
    // If we want to deal with the current mv problem using more streams to balance the workload we still have to consider the race condition (updates the same solution entry simultaneously from multiple processors) if the same row index is in more than one stream!
    for (level=0;level<number_of_levels-1;level++){// the last block-column are just zeroes
        // compute the number of warps per block needed. As before, we prioritize the distribution over the blocks.
        used_warps_per_block = (int) ceil(nWarps/(nSMs*64.0)); // for the gpu we are actually using (computer capability 8.0) 64 is the max number of warps per block
        // matrix vector multiplication over all levels: b_1 = f - B t. Here we exploited the fact that b_i == rhs, and that the function will subtract (alpha = -1) to b_i the result of B t.
        B_column_based_mv(sequence_dataset, cumulative_points, number_of_levels, level, -1, solution, beta_i, mu, nu, wendland_coefficients[0], wendland_coefficients[1], used_warps_per_block, (int) ceil(((double) nWarps)/used_warps_per_block));
    } // end of distributed matrix vector multiplication
    // finished first iteration of the Jacobi solution.
    /// further Jacobi iterations
    for (int counter=1;counter<number_of_levels;counter++){
        // set to 0 the entries of the solution vector, this is done at the start of every iteration to properly store the result of distributed cg
        memset(solution, 0, cumulative_points[number_of_levels]*VALUES_DIM*sizeof(double));
        /// distributed cg routine, step 1: find out how to distribute the warps
        memset(flow_index, 0, number_of_levels * sizeof(short int));//set to 0 the flow_indexes
        level = number_of_levels-2; // the result of the cg of the last block is meaningless since it will go against a 0 matrix in the step of the matrix vector multiplication.
        // we do not need to redefine the variable needed to store the info on the findings used the in the function "distributed_cg", we simply overwrite the old results. Even the level_counter is already 0.
        do {// in the following 32 is the warp_size!
            // set initial resources, the higher the level the higher the priority (to larger sets to smaller), so given a level and estimated the resources and the time needed we try to concurrently solve smaller systems
            available_warps=nWarps;// initial warps resources
            inner_level = level; // highest unsolved level
            // estimate the "computational time" of level with nWarps
            time_scheduled_for_main = (int) ceil((cumulative_points[inner_level+1]-cumulative_points[inner_level])/(32.0*available_warps)); // here (an in the following estimations) we assume that the solution time is linear over the number of points on level
            if (!flow_index[level]){ // check whether the level is still to be solved
                do {
                    // estimate the "computational time" of inner_level with available_warps
                    time_subscheduled = (int) ceil((cumulative_points[inner_level+1]-cumulative_points[inner_level])/(32.0*available_warps));
                    // check whether the time estimation is smaller that the highest level estimate and the inner_level is still to be solved
                    if (time_subscheduled <= time_scheduled_for_main && !flow_index[inner_level]) {
                        // add to the to do list of levels to compute concurrently the inner_level
                        levels_to_do[level_counter]=inner_level;
                        // update the available warps, not based on what is needed but on what we actually use (this quantities may differ since warps are effectively used only when the workload is correctly distributed on the SMs).
                        needed_warps = ceil((cumulative_points[inner_level+1]-cumulative_points[inner_level])/(32.0*time_scheduled_for_main));
                        // compute the number of warps per block needed. Since the aim is a clever resources distribution we prioritize the distribution over the blocks, i.e. with the need of 64 warps we have 64 blocks with 1 warp rather than 1 block with 64 warps
                        used_warps_per_block = (int) ceil(needed_warps/(nSMs*64));
                        // given the number of used_warps_per_block we compute the number of blocks that we need
                        used_blocks = (int) ceil(needed_warps/used_warps_per_block);
                        // remove from the available_warps the warps that we will deploy for this level
                        available_warps -= used_blocks*used_warps_per_block;
                        // add the needed warps and blocks for this level to the list:
                        warps_per_block_needed_at_level[level_counter]=used_warps_per_block;
                        blocks_needed_at_level[level_counter]=used_blocks;
                        // update the index to notify that this level should not be considered anymore
                        flow_index[inner_level]=1;
                        // increase the level_counter and reduce the inner_level to look if other levels can still being executed concurrently
                        level_counter++;
                        inner_level--;
                    } else {inner_level--;} // if the level either do not fit (its expected time is high) or is already solved go further down.
                    // continue the inner_level loop until either we run out of available_warps or all levels were investigated
                } while (available_warps > 0 && inner_level >=0); // end do while
                /// distributed cg routine, step 2: compute the chosen levels.
                // this time the result is stored in the temp_vector (t): solve A_level (t)_level = (b_i)_level for the chosen levels.
                concurrent_cgA(sequence_dataset,cumulative_points, beta_i, eps, solution, levels_to_do, warps_per_block_needed_at_level, blocks_needed_at_level, nSMs, level_counter, streams, mu, nu, wendland_coefficients[0], wendland_coefficients[1]);
            } // end if (we dealt with the higher unsolved level)
            level--; // decrease the highest unsolved level
            level_counter=0; // set to 0 the number of levels to be executed concurrently
        } while (level >= 0); // end distributed cg routine (we went through until there isn't an unsolved level)

        // b_ip1 = f - T b_i. We already exploited the previous iteration result (beta_i) to solve the cg, therefore, we overwrite on it the rhs and we set it to store the result of the new iteration.
        cblas_dcopy(cumulative_points[number_of_levels], rhs, 1, beta_i, 1);

        /// distributed matrix vector multiplication routine: for every block-column compute the matrix vector multiplication and update the result.
        for (level=0;level<number_of_levels-1;level++){// in the last block-column there are just zeroes
            // compute the number of warps per block needed. As before, we prioritize the distribution over the blocks.
            used_warps_per_block = (int) ceil(nWarps/(nSMs*64.0)); // for the gpu we are actually using (computer capability 8.0) 64 is the max number of warps per block
            // matrix vector multiplication over all levels: b_i = f - B t. Here we exploited the fact that b_i == rhs, and that the function will subtract (alpha = -1) to b_i the result of B t.
            B_column_based_mv(sequence_dataset, cumulative_points, number_of_levels, level, -1, solution, beta_i, mu, nu, wendland_coefficients[0], wendland_coefficients[1], used_warps_per_block, (int) ceil(nWarps/used_warps_per_block));
        } // end of distributed matrix vector multiplication
        // now beta_ip1 has the value of the new iteration (?)
    } // end for level
    /// end of the Jacobi solution
    // Free memory within the host
    free(rhs);

    // Now we have solved the Jacobi system T beta = f (stored in beta_i).
    // now we have to solve D_A alpha = beta (that we will store in solution).
    /// Again distributed cg routine, step 1: find out how to distribute the warps
    //set to 0 the flow_indexes (all levels still to be computed)
    memset(flow_index, 0, number_of_levels * sizeof(short int));
    // set the solution to 0
    memset(solution, 0, cumulative_points[number_of_levels]*VALUES_DIM*sizeof(double));
    level = number_of_levels-1; // the result of the cg of the last block is meaningful this time!
    // we do not need to redefine the variable needed to store the info on the findings used the in the function "distributed_cg", we simply overwrite the old results. Even the level_counter is already 0.
    do {// in the following 32 is the warp_size!
        // set initial resources, the higher the level the higher the priority (to larger sets to smaller), so given a level and estimated the resources and the time needed we try to concurrently solve smaller systems
        available_warps=nWarps; // initial warps resources
        inner_level = level; // highest unsolved level
        // estimate the "computational time" of level with nWarps
        time_scheduled_for_main = (int) ceil((cumulative_points[inner_level+1]-cumulative_points[inner_level])/(32.0*available_warps)); // here (an in the following estimations) we assume that the solution time is linear over the number of points on level
        if (!flow_index[level]){ // check whether the level is still to be solved
            do {
                // estimate the "computational time" of inner_level with available_warps
                time_subscheduled = (int) ceil((cumulative_points[inner_level+1]-cumulative_points[inner_level])/(32.0*available_warps));
                // check whether the time estimation is smaller that the highest level estimate and the inner_level is still to be solved
                if (time_subscheduled <= time_scheduled_for_main && !flow_index[inner_level]) {
                    // add to the to do list of levels to compute concurrently the inner_level
                    levels_to_do[level_counter]=inner_level;
                    // update the available warps, not based on what is needed but on what we actually use (this quantities may differ since warps are effectively used only when the workload is correctly distributed on the SMs).
                    needed_warps = ceil((cumulative_points[inner_level+1]-cumulative_points[inner_level])/(32.0*time_scheduled_for_main));
                    // compute the number of warps per block needed. Since the aim is a clever resources distribution we prioritize the distribution over the blocks, i.e. with the need of 64 warps we have 64 blocks with 1 warp rather than 1 block with 64 warps
                    used_warps_per_block = (int) ceil(needed_warps/(nSMs*64));
                    // given the number of used_warps_per_block we compute the number of blocks that we need
                    used_blocks = (int) ceil(needed_warps/used_warps_per_block);
                    // remove from the available_warps the warps that we will deploy for this level
                    available_warps -= used_blocks*used_warps_per_block;
                    // add the needed warps and blocks for this level to the list:
                    warps_per_block_needed_at_level[level_counter]=used_warps_per_block;
                    blocks_needed_at_level[level_counter]=used_blocks;
                    // update the index to notify that this level should not be considered anymore
                    flow_index[inner_level]=1;
                    // increase the level_counter and reduce the inner_level to look if other levels can still being executed concurrently
                    level_counter++;
                    inner_level--;
                } else {inner_level--;}// if the level either do not fit (its expected time is high) or is already solved go further down.
                // continue the inner_level loop until either we run out of available_warps or all levels were investigated
            } while (available_warps > 0 && inner_level >=0); // end do while
            /// distributed cg routine, step 2: compute the chosen levels.
            // this time the result is stored in beta_i: solve A_level (solution)_level = (beta_i)_level for the chosen levels.
            concurrent_cgA(sequence_dataset,cumulative_points, beta_i, eps, solution, levels_to_do, warps_per_block_needed_at_level, blocks_needed_at_level, nSMs, level_counter, streams, mu, nu, wendland_coefficients[0], wendland_coefficients[1]);
        } // end if (we dealt with the higher unsolved level)
        level--; // decrease the highest unsolved level
        level_counter=0; // set to 0 the number of levels to be executed concurrently
    } while (level >= 0); // end of the last distributed cg routine

    // free resources that are not needed anymore
    free(beta_i);
    for (level = 0; level < number_of_levels; level++) {
        cudaStreamDestroy(streams[level]);
    }
    // free memory connected to the cg concurrent solution
    free(streams);
    free(flow_index);
    free(warps_per_block_needed_at_level);
    free(blocks_needed_at_level);
    free(levels_to_do);
}

#endif //PARALLEL_MULTISCALE_APPROXIMATION_ITERATIVE_PARALLEL_METHODS_CUH
