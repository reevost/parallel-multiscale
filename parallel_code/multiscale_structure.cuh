#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "macros.h"

#ifndef PARALLEL_MULTISCALE_APPROXIMATION_MULTISCALE_STRUCTURE_CUH
#define PARALLEL_MULTISCALE_APPROXIMATION_MULTISCALE_STRUCTURE_CUH

double min_array(double * start_pointer,  int len){
    double min_left, min_right;
    if (len == 1){
        return *start_pointer;
    }
    else{
        min_left = min_array(start_pointer, len/2);
        min_right = min_array(start_pointer+len/2, len-len/2);
        if (min_left<min_right) {
            return min_left;
        }
        else{
            return min_right;
        }
    }
}

double max_array(double * start_pointer,  int len){
    double max_left, max_right;
    if (len == 1){
        return *start_pointer;
    }
    else{
        max_left = max_array(start_pointer, len/2);
        max_right = max_array(start_pointer+len/2, len-len/2);
        if (max_left>max_right) {
            return max_left;
        }
        else{
            return max_right;
        }
    }
}

__global__ void partial_separation_distance(double * points, unsigned int number_of_points, double * separation_list_pointer){
    double minimum = 2;
    int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = gridDim.x*blockDim.x;

    for (int i=thread_id; i<number_of_points-1; i+=stride){
        for (int j=i+1; j<number_of_points; j++) {
            double p_dist=0;
            for (int d=0; d < POINTS_DIM; d++){
                p_dist += pow(points[i + d * (number_of_points)] - points[j + d * (number_of_points)], 2);
            }
            // printf("%d, %d, %f\n",i, j, p_dist);
            if (p_dist<minimum){
                minimum = p_dist;
            }
        }
    }
    separation_list_pointer[thread_id] = minimum;
}

double separation_distance(double * points, unsigned int number_of_points, int nThreads, int nBlocks){
    // initialization step
    double * d_points, * d_separation_list, * h_separation_list;
    if (nBlocks*nThreads > number_of_points) {
        nThreads = (number_of_points + nBlocks - 1) / nBlocks;
        if (nThreads==1) {nBlocks = number_of_points;}
    }
    h_separation_list = (double *) calloc(nBlocks*nThreads, sizeof(double));
    cudaMalloc(&d_points, number_of_points*(POINTS_DIM+VALUES_DIM)*sizeof(double));
    cudaMalloc(&d_separation_list, nBlocks*nThreads*sizeof(double));
    cudaMemcpy(d_points, points, number_of_points*(POINTS_DIM+VALUES_DIM)*sizeof(double), cudaMemcpyHostToDevice);
    // computation step
    partial_separation_distance<<<nBlocks, nThreads>>>(d_points, number_of_points, d_separation_list);
    cudaDeviceSynchronize(); // synchronize the threads
    cudaMemcpy(h_separation_list, d_separation_list, nBlocks*nThreads*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_points);
    cudaFree(d_separation_list);
    double minimum = min_array(h_separation_list, nBlocks*nThreads);
    free(h_separation_list);
    return sqrt(minimum)/2;
}

__global__ void partial_fill_distance(double * points, unsigned int number_of_points, double threshold, double * min_vector, int * partial_temp_points, double * fill_list_pointer){
    int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = gridDim.x*blockDim.x;
    //set up the first value of the grid
    double temp_grid_point[POINTS_DIM]; for (int i = 0; i < POINTS_DIM; ++i) {temp_grid_point[i]=min_vector[i];}
    double maximum = 0;
    for (int temp_i=0; temp_i < partial_temp_points[POINTS_DIM]; temp_i++) {
        // test the grid point generator: printf("[ "); for (int d=0; d<dim_; d++) {printf("%f ", temp_grid_point[d]);} printf("]\n");
        double minimum = 2; // should be changed to the maximum possible value
        for (int p_ind = thread_id; p_ind < number_of_points; p_ind += stride) {
            double p_dist=0;
            for (int p_dim = 0; p_dim < POINTS_DIM; p_dim++) {
                p_dist += pow(temp_grid_point[p_dim] - points[p_ind + p_dim * (number_of_points)], 2);
            }
            if (p_dist < minimum) {
                minimum = p_dist;
            }
        }
        if (minimum > maximum) {
            // printf("%f, %f -> %f\n", temp_grid_point[0], temp_grid_point[1], minimum);
            maximum = minimum;
        }

        // increase the grid by one step
        temp_grid_point[0] += threshold;
        for (int d = 1; d < POINTS_DIM; d++) {
            if ((temp_i + 1) % partial_temp_points[d] == 0) {
                temp_grid_point[d] += threshold;
                if (d != 0) {
                    temp_grid_point[d - 1] = min_vector[d - 1];
                }
            }
        } // end increase
    }
    fill_list_pointer[thread_id] = maximum;
}

double fill_distance(double * points, unsigned int number_of_points, double threshold, int nThreads, int nBlocks){
    //initialization step
    double * d_points, * d_fill_list, * h_fill_list;
    if (nBlocks*nThreads > number_of_points) {
        nThreads = (number_of_points + nBlocks - 1) / nBlocks;
        if (nThreads==1) {nBlocks = number_of_points;}
    }
    h_fill_list = (double *) calloc(nBlocks*nThreads, sizeof(double));
    cudaMalloc(&d_points, number_of_points*(POINTS_DIM+VALUES_DIM)*sizeof(double));
    cudaMalloc(&d_fill_list, nBlocks*nThreads*sizeof(double));
    cudaMemcpy(d_points, points, number_of_points*(POINTS_DIM+VALUES_DIM)*sizeof(double), cudaMemcpyHostToDevice);
    double min_vector[POINTS_DIM], max_vector[POINTS_DIM];
    int partial_temp_points[POINTS_DIM+1], total_temp_points = 1;
    //preprocessing
    partial_temp_points[0] = 1;
    for (int d=0; d<POINTS_DIM; d++) {
        min_vector[d] = 0;
        max_vector[d] = 1;
        /* if not in the standard settings [0,1]^d
        min_vector[d] = points[0][d];
        max_vector[d] = points[0][d];
        for (int i=0; i<number_of_points; i++){
            if (points[i][d] < min_vector[d]){
                min_vector[d]= points[i][d];
            }
            else if (points[i][d] > max_vector[d]){
                max_vector[d] = points[i][d];
            }
        }*/
        total_temp_points *= floor((max_vector[d]-min_vector[d])/threshold+1);
        partial_temp_points[d+1] = total_temp_points;
    }
    // computation step
    double * d_min_vector; int * d_partial_temp_points;
    cudaMalloc(&d_min_vector, POINTS_DIM * sizeof(double)); cudaMalloc(&d_partial_temp_points, (POINTS_DIM+1) * sizeof(int));
    cudaMemcpy(d_min_vector, min_vector, POINTS_DIM * sizeof(double), cudaMemcpyHostToDevice); cudaMemcpy(d_partial_temp_points, partial_temp_points, (POINTS_DIM+1) * sizeof(int), cudaMemcpyHostToDevice);

    partial_fill_distance<<<nBlocks, nThreads>>>(d_points, number_of_points, threshold, d_min_vector, d_partial_temp_points, d_fill_list);
    cudaDeviceSynchronize(); // synchronize the threads
    cudaMemcpy(h_fill_list, d_fill_list, nBlocks*nThreads*sizeof(double), cudaMemcpyDeviceToHost);
    // printf("partial fill distance: %f\n", d_fill_list[0]);
    cudaFree(d_points); cudaFree(d_fill_list);
    cudaFree(d_min_vector); cudaFree(d_partial_temp_points);
    double maximum = max_array(h_fill_list, nBlocks*nThreads);
    free(h_fill_list);
    return sqrt(maximum);
}

__device__ void ranking_strict(double * array, unsigned int n, double value, int * position){
    /// found the rank of value on the first n elements of the array.
    /// array: pointer to the first element of a sorted array.
    /// n: length of the array.
    /// value: value to be ranked on array.
    /// position: index for the binary search. It will contain the rank of the value on array. Its starting value should be set as n/2.

    /// complexity notes:
    /// space: O(1)
    /// time: O(log2(n))
    while (n > 1){ // we search until the search is restricted to a sub-array of length up to 1
        if (value > array[*position]){
            // value is grater than the element in the current position
            (*position)+=(n+3)/4; // increase position so that is always in the middle of our search sub-array
            n=(n-1)/2; // halve sub-array length
        } else {
            // value is smaller or equal than the element in the current position
            (*position)-=(n+2)/4; // decrease position so that is always in the middle of our search sub-array
            n=n/2; // halve sub-array length
        }
    }
    // here either n == 1 or n == 0. In the first case we have to understand if our value come before or after the element in the sub-array. Otherwise, we already have found the rank.
    if (value > array[*position] && n) {
        // n==1 case with our value grater than the element in current position
        (*position)++;
    }
}

__device__ void ranking_equal(double * array, unsigned int n, double value, int * position){
    /// found the inclusive rank of value on the first n elements of the array: i.e. count the number of elements smaller or equal than value on given array
    /// array: pointer to the first element of a sorted array.
    /// n: length of the array.
    /// value: value to be ranked on array.
    /// position: index for the binary search. It will contain the rank of the value on array. Its starting value should be set as n/2.

    /// complexity notes:
    /// space: O(1)
    /// time: O(log2(n))
    while (n > 1){ // we search until the search is restricted to a sub-array of length up to 1
        if (value >= array[*position]){
            // value is grater or equal than the element in the current position
            (*position)+=(n+3)/4; // increase position so that is always in the middle of our search sub-array
            n=(n-1)/2; // halve sub-array length
        } else {
            // value is smaller than the element in the current position
            (*position)-=(n+2)/4; // decrease position so that is always in the middle of our search sub-array
            n=n/2; // halve sub-array length
        }
    }
    // here either n == 1 or n == 0. In the first case we have to understand if our value come before or after the element in the sub-array. Otherwise, we already have found the rank.
    if (value >= array[*position] && n) {
        // n==1 case with our value grater or equal than the element in current position
        (*position)++;
    }
}

__global__ void parallel_multi_block_sort(double * array, unsigned int * indices, unsigned int * lengths, const unsigned int array_len, const unsigned int number_of_points, unsigned int axis){
    /// Sort gridDim sub-arrays (parametrized by indices and lengths, of size up to blockDim) in parallel until it sorts all sub-arrays parametrized by indices and lengths.
    /// array: a pointer to the first element of the full point array (the pointer to the to-be tree).
    /// indices: pointer to the indices array, which contain starting indices (with respect to array) of the sub-arrays to sort.
    /// lengths: pointer to the lengths array, which contain the lengths of the sub-arrays to sort.
    /// array_len: number of sub-arrays to be sorted.
    /// total_number_of_points: number of elements between coordinates of the same point (see data allocation structure in main). Can also be seen as the total number of points stored between all levels.
    /// axis: axis with respect to the sorting happen.

    /// complexity notes:
    /// space on device: O(1):
    /// time: O(N/gridDim*log2^2(blockDim))  N=number_of_points, n=local_number_of_points
    // every block will sort one sub-array
    for (int block = blockIdx.x; block<array_len; block+=gridDim.x) { // if the array is too long we need that the blocks perform multiple tasks
        // first parallel batcher sort, which sort every chunk of blockDim elements.
        int t = ceil(log2f(lengths[block]));
        int p = 1<<(t-1); // p, q, r, d are parameters of the batcher function I kept them as in the original paper
        while (p > 0){
            int q = 1<<(t-1);
            int r = 0;
            int d = p;
            while (d > 0){
                if (threadIdx.x < lengths[block]-d && (threadIdx.x & p) == r && array[indices[block]+threadIdx.x+number_of_points*axis] > array[indices[block]+threadIdx.x+number_of_points*axis+d]){
                    for (int temp_dim=0; temp_dim<POINTS_DIM+VALUES_DIM; temp_dim++){
                        double temp_double = array[indices[block]+threadIdx.x+number_of_points*temp_dim];
                        array[indices[block]+threadIdx.x+number_of_points*temp_dim] = array[indices[block]+threadIdx.x+number_of_points*temp_dim+d];
                        array[indices[block]+threadIdx.x+number_of_points*temp_dim+d] = temp_double;
                    }
                }
                __syncthreads();
                d = q-p;
                q = q/2;
                r = p;
            }
            p = p/2;
        } // end parallel batcher sort
        // then, update the indices and length of the sub array to proceed deeper in the tree construction. Indeed, this kernel will be launched until the tree is built.
        if (threadIdx.x == 0){ // we to go over this just once for every block!
            indices[array_len+block] = indices[block]+lengths[block]/2+1;
            lengths[array_len+block] = (lengths[block]-1)/2;
            lengths[block] /= 2;
        }
    }
}

__global__ void parallel_mergesort_block_sort(double * array, const unsigned int local_number_of_points, const unsigned int total_number_of_points, unsigned int axis){
    /// Sort chunks of size BlockDim. A merge step is needed to have the array sorted.
    /// array: pointer to the first element of the array to sort in chunks.
    /// local_number_of_points: length of the array to sort in chunks.
    /// total_number_of_points: number of elements between coordinates of the same point (see data allocation structure in main). Can also be seen as the total number of points stored between all levels.
    /// axis: axis with respect to the sorting happen.

    /// complexity notes:
    /// space on device: O(1):
    /// time: O(N/gridDim*log2^2(blockDim))   N=number_of_points, n=local_number_of_points
    int block_size = blockDim.x; // get the number of threads within a block
    double chunks_on_points = local_number_of_points/(double)block_size; // compute how many chunks need to be sorted

    // Parallel batcher sort, which sort every chunk of blockDim elements. Every block will sort on chunk.
    for (int block = blockIdx.x; block<chunks_on_points; block+=gridDim.x) { // if the array is too long (we don't have enough block in our grid) we need that the blocks perform multiple tasks
        int n = (block+1)*block_size < local_number_of_points ? block_size : local_number_of_points-block*block_size; // number of elements in the chunk of data that has to be sorted with parallel batcher
        int t = ceil(log2f(n));
        int p = 1<<(t-1); // p, q, r, d are parameters of the batcher function I kept them as in the original paper
        while (p > 0){
            int q = 1<<(t-1);
            int r = 0;
            int d = p;
            while (d > 0){
                if (threadIdx.x < n-d && (threadIdx.x & p) == r && array[threadIdx.x+block*block_size+total_number_of_points*axis] > array[threadIdx.x+block*block_size+total_number_of_points*axis+d]){
                    for (int temp_dim=0; temp_dim<POINTS_DIM+VALUES_DIM; temp_dim++){
                        double temp_double = array[threadIdx.x+block*block_size+total_number_of_points*temp_dim];
                        array[threadIdx.x+block*block_size+total_number_of_points*temp_dim] = array[threadIdx.x+block*block_size+total_number_of_points*temp_dim+d];
                        array[threadIdx.x+block*block_size+total_number_of_points*temp_dim+d] = temp_double;
                    }
                }
                __syncthreads();
                d = q-p;
                q = q/2;
                r = p;
            }
            p = p/2;
        }
    } // end parallel batcher sort
}

__global__ void parallel_mergesort_merge(double * array, const unsigned int local_number_of_points, const unsigned int total_number_of_points, double * temp_arr, unsigned int pow2_depth, const unsigned int axis, double pair_couples){
    /// follow up of parallel_mergesort_block_sort, merge sorted couples at a given depth. should be called ceil(log2(number_of_chunks)) times to completely sort the array.
    /// The merge is not in place, the result is stored on another array. the merge is achieved using the ranking technique (the rank of an element is equal to the number of entries smaller than him in a sorted array),
    /// where we compute for every point in both array their rank with respect to the other array, this is summed with their index and mark their index on the merged array.
    // parallel merge of piece [2^d*blockDim*(block_id), 2^d*blockDim*(block_id)+2^(d-1)*blockDim-1] and [2^d*blockDim*(block_id)+2^(d-1)*blockDim, min(2^d*blockDim*(block_id+1), n)]
    /// array: pointer to the first element of the array to sort in chunks.
    /// local_number_of_points: length of the array sorted in chunks to merge.
    /// total_number_of_points: number of elements between coordinates of the same point (see data allocation structure in main). Can also be seen as the total number of points stored between all levels.
    /// temp_arr: pointer to the array where the merge will be stored.
    /// pow2_depth: 2^depth, tell us how many chunks need to merge (and consequently their max size). Would be equal to pass depth and then compute the 2-power.
    /// axis: axis with respect to the sorting happen.
    /// pair_couples: how many chunks need the merge.

    /// complexity notes:
    /// space on device: O(1):
    /// time: O(N/gridDim*log2(n))  N=number_of_points, n=local_number_of_points
    int block_size = blockDim.x; // get the number of threads within a block
    double blocks_on_points = local_number_of_points/(double)block_size; // compute how many blocks are needed ot work on the full array, which corresponds to the original amount of chucks to be merged
    for (int block = blockIdx.x; block<blocks_on_points; block+=gridDim.x) { // if the array is too long we need that the blocks perform multiple tasks
        // each block is assigned to one original chunk (with larger chunks, we may have many blocks associated with the same chunk)
        if (threadIdx.x+block*block_size<local_number_of_points) { // thread allowed (i.e. in the range) (indeed if local_number_of_points isn't a multiple of blockDim the last chunk has fewer elements
            int pair_block_id = pow2_depth*(block/pow2_depth)+pow2_depth-2*pow2_depth*((block/pow2_depth)%2); // starting index for the paired chunk (that needs to be merged with the chunk where this block is) at given depth
            int pair_block_size = pow2_depth*block_size <= local_number_of_points-pair_block_id*block_size ? pow2_depth*block_size : local_number_of_points%(pow2_depth*block_size); // number of elements in the paired chunk
            if (pair_block_id<(pair_couples/2)*(2*pow2_depth)){ // chunk can merge (i.e. is paired to another chunk)
                // compute the new index of the point associated with this thread, meaning current index+rank on the pair chunk
                int start_chunk = (2*pow2_depth)*(block/(2*pow2_depth)); // index of the first element of the chunk within the array (since we might have many blocks within the same chunk, but their "current index" needs to take into account the chunk not the block)
                // evaluate the rank on the paired chunk
                int rank = pair_block_size/2; // since the array is sorted, the rank search is done in log2(n) operations. rank will initially hold the initial value of the search
                if (block < pair_block_id) {
                    // use one rank function
                    ranking_equal(&array[pair_block_id*block_size+total_number_of_points*axis], (int) pair_block_size, array[threadIdx.x+block*block_size+total_number_of_points*axis], &rank); // ranking of (threadIdx, blockIdx) in the chunk
                } else {
                    // use the other one
                    ranking_strict(&array[pair_block_id*block_size+total_number_of_points*axis], (int) pair_block_size, array[threadIdx.x+block*block_size+total_number_of_points*axis], &rank); // ranking of (threadIdx, blockIdx) in the chunk
                }
                /// comment: the employment of two ranking function is motivated by the possible presence of duplicate values.
                /// more details can be found in the code description on the article
                // store in the new array the value (of threadIdx, blockIdx)
                for (int temp_dim=0; temp_dim<POINTS_DIM+VALUES_DIM; temp_dim++){ // both array and temp_arr support the same structure for storing points
                    temp_arr[(threadIdx.x+block*block_size)%(block_size*pow2_depth)+start_chunk*block_size+total_number_of_points*temp_dim+rank] = array[threadIdx.x+block*block_size+total_number_of_points*temp_dim];
                }
            } else { // meaning, there's no pair chunk, we have just to copy the values in temp_arr
                for (int temp_dim=0; temp_dim<POINTS_DIM+VALUES_DIM; temp_dim++){
                    temp_arr[threadIdx.x+block*block_size+total_number_of_points*temp_dim] = array[threadIdx.x+block*block_size+total_number_of_points*temp_dim];
                }
            }
        }
    }
}

void ParallelKDTreeStructure(double * data, const unsigned int number_of_points, const int deviceId, const unsigned int warp_per_block){
    /// build a kd-tree over a given dataset
    /// data: pointer to the first double of the point set, i.e. x_p0. The points must be stored first level-wise and secondly dimension-wise (see main)
    /// number_of_points: the cumulative size of the points, i.e. N = sum N(level).
    /// deviceId: id associated to the device over which the parallel processes will run
    /// warp_per_block: number of warps that should be deployed on every block

    /// complexity notes:
    /// space on device: O(N): 2dN doubles + 2N ints.
    /// time: O(N): (N/p)log2(N/blockDim)[log2^2(blockDim)+log2^2(N/blockDim)] ~ Nlog2^3(N)/p
    // device setup, gathering basic information
    int nSMs;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    cudaDeviceGetAttribute(&nSMs, cudaDevAttrMultiProcessorCount, deviceId);
    // cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, deviceId); we assume that warp_size = 32.
    // we take as parameter the number of warps per block and based on that we compute the block size based on the occupancy
    int thread_per_block = 32*warp_per_block; // definition and assignment of the number of thread per block
    int block_counter_bucket, block_counter_merge, block_counter_multi; // definition of different block sizes based on the purposes.

    // These variables are used to convert occupancy to warps. Given the kernel to launch and the number of threads per block, compute the maximum number of blocks allowed to run.
    // since the result is for a single SM we multiply the result for their cardinality
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&block_counter_bucket, parallel_mergesort_block_sort, thread_per_block, 0); block_counter_bucket *= nSMs;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&block_counter_merge, parallel_mergesort_merge, thread_per_block, 0); block_counter_merge *= nSMs;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&block_counter_multi, parallel_multi_block_sort, thread_per_block, 0); block_counter_multi *= nSMs;
    //printf("Occupancy: thread_per_block = %d, block_counter_bucket = %d, block_counter_merge = %d, block_counter_multi = %d\n", thread_per_block, block_counter_bucket, block_counter_merge, block_counter_multi);

    double * clone_dev, * data_dev; //definition of data and a copy (clone)
    CUDA_ERROR_CHECK(cudaMalloc(&clone_dev, number_of_points * (POINTS_DIM+VALUES_DIM) * sizeof(double))); //allocation of data
    CUDA_ERROR_CHECK(cudaMalloc(&data_dev, number_of_points * (POINTS_DIM+VALUES_DIM) * sizeof(double))); //allocation of copy (clone)
    CUDA_ERROR_CHECK(cudaMemcpy(data_dev, data, number_of_points * (POINTS_DIM+VALUES_DIM) * sizeof(double), cudaMemcpyHostToDevice)); //copy of the data from host to device

    // definition and allocation for indices and lengths we need to keep track of during the tree construction process.
    // Since we keep track for every sub array of starting index and length, we need space equal to the closest power of 2 to our dataset
    unsigned int * indices; cudaMallocManaged(&indices, (1<<(int) (floor(log2f(number_of_points)))) * sizeof(unsigned int)); indices[0] = 0;
    unsigned int * lengths; cudaMallocManaged(&lengths, (1<<(int) (floor(log2f(number_of_points)))) * sizeof(unsigned int)); lengths[0] = number_of_points;

    for (int depth=0; depth<floor(log2f(number_of_points)); depth++){ // how deep we are on the tree: at given depth 2^depth arrays need to be sorted.
        if (lengths[0] > thread_per_block){ // some pieces are still greater than the threads_per_block (therefore we cannot parallely sort the sub arrays). Here we exploit the fact that the left most block has always the highest number of points.
            for (unsigned int i=0; i<1<<depth; i++) { // Sort the data from X[indices[i]] to X[indices[i] + length[i]] with respect to the depth-th coordinate.
                // First, we sort chunks of size thread_per_block within the sub array
                parallel_mergesort_block_sort<<<block_counter_bucket, thread_per_block>>>(&data_dev[indices[i]], lengths[i], number_of_points, depth % (POINTS_DIM));
            }
            CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Synchronize the threads
            for (unsigned int i=0; i<1<<depth; i++) { // at every depth we have to sort "i" sub-arrays
                // Second, we merge together the sorted chunks
                double blocks_on_points = lengths[i]/(double)thread_per_block; // #chunks to merge back
                double q = blocks_on_points; // parameter to keep track how to merge the chunks
                for (unsigned int m_depth=0; m_depth<log2(blocks_on_points); m_depth++){ // at each merge_depth we merge sorted arrays with thread_per_block*2^{merge_depth} points (or less for the last block).
                    // unfortunately the merge cannot be done in place, therefore we have "clone" to temporary store data.
                    if (m_depth%2) { // with odd m_depth we copy from clone to data
                        parallel_mergesort_merge<<<block_counter_merge, thread_per_block>>>(&clone_dev[indices[i]], lengths[i], number_of_points, &data_dev[indices[i]], 1U<<m_depth, depth%(POINTS_DIM), q);
                    } else { // with even m_depth we copy from data to clone
                        parallel_mergesort_merge<<<block_counter_merge, thread_per_block>>>(&data_dev[indices[i]], lengths[i], number_of_points, &clone_dev[indices[i]], 1U<<m_depth, depth%(POINTS_DIM), q);
                    }
                    CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Synchronize the threads
                    q = ceil(q/2); // chunks to merge at next step
                }
                if ((int)ceil(log2(blocks_on_points))%2) { //final value on clone array
                    for (int temp_dim=0; temp_dim<POINTS_DIM+VALUES_DIM; temp_dim++){ //copy them back to data_dev
                        CUDA_ERROR_CHECK(cudaMemcpy(&data_dev[indices[i]+number_of_points*temp_dim], &clone_dev[indices[i]+number_of_points*temp_dim], lengths[i] * sizeof(double), cudaMemcpyDeviceToDevice));
                    }
                }
                // update the indices and length of the sub array to proceed deeper in the tree construction
                // here we for every "old" sub array we create 2: one with the same index but half-length, the other that start when the previous end and last till the end of the "old" sub array.
                // The point "in the middle" is in neither of them since is the node of the tree.
                indices[(1<<depth)+i] = indices[i]+lengths[i]/2+1; // new sub array index
                lengths[(1<<depth)+i] = (lengths[i]-1)/2; // new sub array length
                lengths[i] /= 2; // update sub array length
            } // endfor inner loop
        } /* endif large blocks */else { // all blocks have less than thread_per_block points. Therefore, we can sort them with batcher sort in parallel
            parallel_multi_block_sort<<<block_counter_multi, thread_per_block>>>(data_dev, indices, lengths, 1<<depth, number_of_points, depth%(POINTS_DIM));
            CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Synchronize the threads
        } // endif small block sort
    } // endfor outer-depth loop -> data kd-tree organized
    CUDA_ERROR_CHECK(cudaMemcpy(data, data_dev, number_of_points * (POINTS_DIM+VALUES_DIM) * sizeof(double), cudaMemcpyDeviceToHost)); //copy back the built tree on host
    // free allocated memory
    CUDA_ERROR_CHECK(cudaFree(indices));
    CUDA_ERROR_CHECK(cudaFree(lengths));
    CUDA_ERROR_CHECK(cudaFree(data_dev));
    CUDA_ERROR_CHECK(cudaFree(clone_dev));
}

__global__ void computeGridPoints2d(double * points, double grid_size, unsigned int number_of_points_on_axis, unsigned int number_of_points) {
    /// compute (in parallel) on frank function evaluated on (mu^{1+level}+1)^2 equispaced points in [0,1]^2
    /// points: pointer to a suitable location where the point can be stored
    /// level: level at which the grid is built
    /// grid size: distance between two points on the same axis
    /// number_of_points_on_axis: number of points on the same axis
    /// total_number_of_points: overall number of points, needed to properly store the values accordingly to our structure (see data allocation on main)
    // we expect a 2d thread and block structure
    for (int i=blockIdx.x * blockDim.x + threadIdx.x; i<number_of_points_on_axis; i+=blockDim.x*gridDim.x){
        for (int j=blockIdx.y * blockDim.y + threadIdx.y; j<number_of_points_on_axis; j+=blockDim.y*gridDim.y){
            //x
            points[i+number_of_points_on_axis*j] = i*grid_size;
            //y
            points[i+number_of_points_on_axis*j+number_of_points] = j*grid_size;
            //f(x,y)
            points[i+number_of_points_on_axis*j+2*number_of_points] =
                    0.75 * exp(-pow(9 * points[i+number_of_points_on_axis*j] - 2, 2)/4 - pow(9 * points[i+number_of_points_on_axis*j+number_of_points] - 2, 2)/4) +
                    0.75 * exp(-pow(9 * points[i+number_of_points_on_axis*j] + 1, 2)/49 - pow(9 * points[i+number_of_points_on_axis*j+number_of_points] + 1, 2)/49) +
                    0.5 * exp(-pow(9 * points[i+number_of_points_on_axis*j] - 7, 2)/4 - pow(9 * points[i+number_of_points_on_axis*j+number_of_points] - 3, 2)/4) -
                    0.2 * exp(-pow(9 * points[i+number_of_points_on_axis*j] - 4, 2) - pow(9 * points[i+number_of_points_on_axis*j+number_of_points] - 7, 2));
        }
    }
}

__global__ void updateGridPoints2d(double * points, int reference_index, unsigned int number_of_points_on_axis, unsigned int number_of_points) {
    /// compute (in parallel) on frank function evaluated on (mu^{1+level}+1)^2 equispaced points in [0,1]^2
    /// points: pointer to a suitable location where the point can be stored
    /// level: level at which the grid is built
    /// grid size: distance between two points on the same axis
    /// number_of_points_on_axis: number of points on the same axis
    /// total_number_of_points: overall number of points, needed to properly store the values accordingly to our structure (see data allocation on main)
    // we expect a 2d thread and block structure
    for (int i=blockIdx.x * blockDim.x + threadIdx.x; i<number_of_points_on_axis; i+=blockDim.x*gridDim.x){
        for (int j=blockIdx.y * blockDim.y + threadIdx.y; j<number_of_points_on_axis; j+=blockDim.y*gridDim.y){
            double squared_norm = 0;
            for (int d=0; d<POINTS_DIM; d++) { squared_norm += pow(points[i+number_of_points_on_axis*j+d*number_of_points] - points[reference_index+d*number_of_points], 2); }
            //f(x,y)
            points[i+number_of_points_on_axis*j+2*number_of_points] = exp(-sqrt(squared_norm));
        }
    }
}

// for testing purposes
void tree_check(double * zero_tree_pointer, unsigned int number_of_points, double delta) {
    /// check if tree is correctly built counting the delta neighbours
    double squared_norm; double diff_axis; int counter, flag=0;
    // check every point of the tree within the tree
    for (int i=0; i<number_of_points; i++) {
        counter=0;
        // check with n^2 complexity
        for (int j = 0; j < number_of_points; j++) {
            squared_norm = 0;
            for (int d = 0; d < POINTS_DIM; d++) {
                squared_norm += pow(zero_tree_pointer[i+d*number_of_points] - zero_tree_pointer[j+d*number_of_points], 2);
            }
            if (squared_norm < delta * delta) {
                //printf("Entry (%f, %f) is non zero\n", zero_tree_pointer[number_of_points/2], zero_tree_pointer[number_of_points/2+number_of_points]);
                counter++;
            }
        }

        // check with log complexity (tree visit)
        int to_visit_list[30], to_length_list[30], to_axis_list[30];
        // to_axis_list can be replaced using the information on the length of the subtree: axis = (depth-log2(tree_length)-1)%POINTS_DIM /// not sure, double check

        // test the root
        squared_norm = 0;
        for (int d = 0; d < POINTS_DIM; d++) {
            squared_norm += pow(zero_tree_pointer[number_of_points/2+d*number_of_points] - zero_tree_pointer[i+d*number_of_points], 2);
        }
        if (squared_norm < delta * delta) {
            //printf("Entry (%f, %f) is non zero\n", zero_tree_pointer[number_of_points/2], zero_tree_pointer[number_of_points/2+number_of_points]);
            counter--;
        } //else {printf("Entry (%f, %f) is zero\n", zero_tree_pointer[number_of_points/2], zero_tree_pointer[number_of_points/2+number_of_points]);}
        // then add to the waiting list the children of the root since a priori no one can be excluded
        to_visit_list[1] = number_of_points / 2 - (number_of_points + 2) / 4;
        to_length_list[1] = number_of_points / 2;
        to_axis_list[1] = 1;
        to_visit_list[0] = number_of_points / 2 + (number_of_points + 3) / 4;
        to_length_list[0] = (number_of_points - 1) / 2;
        to_axis_list[0] = 1;
        int visit_index = 1;
        while (visit_index >= 0) {
            squared_norm = 0;
            for (int d = 0; d < POINTS_DIM; d++) {
                squared_norm += pow(zero_tree_pointer[to_visit_list[visit_index]+d*number_of_points] - zero_tree_pointer[i+d*number_of_points], 2);
            }
            if (squared_norm < delta * delta) {
                counter--;
                //printf("Entry (%f, %f) is non zero\n", zero_tree_pointer[to_visit_list[visit_index]], zero_tree_pointer[to_visit_list[visit_index]+number_of_points]);
            } //else {printf("Entry (%f, %f) is zero\n", zero_tree_pointer[to_visit_list[visit_index]], zero_tree_pointer[to_visit_list[visit_index]+number_of_points]);}
            if (to_length_list[visit_index] == 1) {// leaf case
                visit_index--;
            } else if (to_length_list[visit_index] == 2) {//node with one child (the left one)
                to_visit_list[visit_index] -= (to_length_list[visit_index] + 2) / 4;
                to_length_list[visit_index] /= 2;
                to_axis_list[visit_index] = (to_axis_list[visit_index] + 1) % POINTS_DIM;
            } else { // inner node
                diff_axis = zero_tree_pointer[to_visit_list[visit_index] + to_axis_list[visit_index] * number_of_points] - zero_tree_pointer[i+to_axis_list[visit_index]*number_of_points];
                if (diff_axis > 0) {// the left child is closer (actually closer is not really correct, more likely to give useful results suits better)
                    // firstly check if also the right child should be explored
                    // indeed, adding the left child  on the current point will alter the values needed to the other child determination
                    // on the other hand, if the right child is not meaningful to visit we have to store the left one over the current point
                    if (diff_axis * diff_axis <= delta * delta) {
                        // yes, we add the right child to the visit list
                        to_visit_list[visit_index + 1] = to_visit_list[visit_index] + (to_length_list[visit_index] + 3) / 4;
                        to_length_list[visit_index + 1] = (to_length_list[visit_index] - 1) / 2;
                        to_axis_list[visit_index + 1] = (to_axis_list[visit_index] + 1) % POINTS_DIM;
                        // and of course the left child
                        to_visit_list[visit_index] -= (to_length_list[visit_index] + 2) / 4;
                        to_length_list[visit_index] = to_length_list[visit_index] / 2;
                        to_axis_list[visit_index] = to_axis_list[visit_index + 1];
                        visit_index++;
                    } else { // otherwise, we add just the left child to the visit list
                        to_visit_list[visit_index] -= (to_length_list[visit_index] + 2) / 4;
                        to_length_list[visit_index] = to_length_list[visit_index] / 2;
                        to_axis_list[visit_index] = (to_axis_list[visit_index] + 1) % POINTS_DIM;
                    } // end if-else which-child-explore
                } else { // the right child is the closer
                    // similarly to the other case, firstly check if also the left child should be explored
                    if (diff_axis * diff_axis <= delta * delta) {
                        // yes, we add the left child to the visit list
                        to_visit_list[visit_index + 1] = to_visit_list[visit_index] - (to_length_list[visit_index] + 2) / 4;
                        to_length_list[visit_index + 1] = to_length_list[visit_index] / 2;
                        to_axis_list[visit_index + 1] = (to_axis_list[visit_index] + 1) % POINTS_DIM;
                        // and of course the right child
                        to_visit_list[visit_index] += (to_length_list[visit_index] + 3) / 4;
                        to_length_list[visit_index] = (to_length_list[visit_index] - 1) / 2;
                        to_axis_list[visit_index] = to_axis_list[visit_index + 1];
                        visit_index++;
                    } else { // otherwise, we add just the right child to the visit list
                        to_visit_list[visit_index] += (to_length_list[visit_index] + 3) / 4;
                        to_length_list[visit_index] = (to_length_list[visit_index] - 1) / 2;
                        to_axis_list[visit_index] = (to_axis_list[visit_index] + 1) % POINTS_DIM;
                    } // end if-else which-child-explore
                } // end if-else closer-to-reference-child
            } //end if-elif-else leaf-double-inner node
        } // end while there are still points to visit
        //if (!counter) {printf("index %d, counter: %d\n", i, counter); flag++;}
    }
    if (flag) {printf("the tree is not correctly built, there are %d misclassified entries\n", flag);} else {printf("The tree is correctly built");}
}

void ParallelMergeSort(double * data, const unsigned int number_of_points, const unsigned int axis, const int deviceId){
    // device setup, gathering basic information
    int nSMs, warp_size;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    cudaDeviceGetAttribute(&nSMs, cudaDevAttrMultiProcessorCount, deviceId);
    cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, deviceId);
    int thread_per_block = warp_size, block_counter; // warp_size is usually 32.

    // These variables are used to convert occupancy to warps
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&block_counter, parallel_mergesort_block_sort, thread_per_block, 0);
    block_counter *= nSMs;
    //printf("Bucket occupancy: thread_per_block = %d, block_counter = %d\n", thread_per_block, block_counter);

    double * clone_dev, * data_dev;
    CUDA_ERROR_CHECK(cudaMalloc(&clone_dev, number_of_points * (POINTS_DIM+VALUES_DIM) * sizeof(double)));
    CUDA_ERROR_CHECK(cudaMalloc(&data_dev, number_of_points * (POINTS_DIM+VALUES_DIM) * sizeof(double)));
    CUDA_ERROR_CHECK(cudaMemcpy(data_dev, data, number_of_points * (POINTS_DIM+VALUES_DIM) * sizeof(double), cudaMemcpyHostToDevice));

    parallel_mergesort_block_sort<<<block_counter, thread_per_block>>>(data_dev, number_of_points, number_of_points, axis);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Synchronize the threads

    // compute merge occupancy
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&block_counter, parallel_mergesort_merge, thread_per_block, 0);
    block_counter *= nSMs;
    //printf("Merge occupancy: thread_per_block = %d, block_counter = %d\n", thread_per_block, block_counter);

    double blocks_on_points = number_of_points/(double)thread_per_block;
    double q = blocks_on_points; printf("merging %d blocks of size %d from %d-array\n", (int) ceil(blocks_on_points), thread_per_block, number_of_points);
    for (int depth=0; depth<log2(blocks_on_points); depth++){
        if (depth%2) {
            //printf("clone: q = %f, depth: %d\n", q, depth);
            parallel_mergesort_merge<<<block_counter, thread_per_block>>>(clone_dev, number_of_points, number_of_points, data_dev, 1U<<depth, axis, q);
        } else {
            //printf("data: q = %f, depth: %d\n", q, depth);
            parallel_mergesort_merge<<<block_counter, thread_per_block>>>(data_dev, number_of_points, number_of_points, clone_dev, 1U<<depth, axis, q);
        }
        CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Synchronize the threads
        q = ceil(q/2);
    }
    if ((int)ceil(log2(blocks_on_points))%2) {
        CUDA_ERROR_CHECK(cudaMemcpy(data, clone_dev, number_of_points * (POINTS_DIM+VALUES_DIM) * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        CUDA_ERROR_CHECK(cudaMemcpy(data, data_dev, number_of_points * (POINTS_DIM+VALUES_DIM) * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_ERROR_CHECK(cudaFree(data_dev));
    CUDA_ERROR_CHECK(cudaFree(clone_dev));
}

__device__ void DeviceOddEvenMergeSort(double * tree, int number_of_points, int axis){ // for testing purposes
    /// Batcher's odd–even mergesort should be applied to only small arrays (up to 1024 entries)
    /// sort the first n entries of the array arr
    //int thread_id = blockIdx.x*blockDim.x+threadIdx.x; //int stride = gridDim.x*blockDim.x;
    int block; int block_size = blockDim.x; double blocks_on_points = number_of_points/(double)block_size;

    // parallel batcher sort, which sort every chunk of blockDim elements.
    for (block = blockIdx.x; block<blocks_on_points; block+=gridDim.x) { // if the array is too long we need that the blocks perform multiple tasks
        int t = ceil(log2f(block_size));
        int p = 1<<(t-1); // p, q, r, d are parameters of the batcher function I kept them as in the original paper
        int n = (block+1)*block_size < number_of_points ? block_size : number_of_points-block*block_size; // number of elements in the chunk of data that has to be sorted with parallel batcher
        while (p > 0){
            int q = 1<<(t-1);
            int r = 0;
            int d = p;
            while (d > 0){
                if (threadIdx.x < n-d && (threadIdx.x & p) == r && tree[threadIdx.x+block*block_size+number_of_points*axis] > tree[threadIdx.x+block*block_size+number_of_points*axis+d]){
                    for (int temp_dim=0; temp_dim<POINTS_DIM+VALUES_DIM; temp_dim++){
                        double temp_double = tree[threadIdx.x+block*block_size+number_of_points*temp_dim];
                        tree[threadIdx.x+block*block_size+number_of_points*temp_dim] = tree[threadIdx.x+block*block_size+number_of_points*temp_dim+d];
                        tree[threadIdx.x+block*block_size+number_of_points*temp_dim+d] = temp_double;
                    }
                }
                __syncthreads();
                d = q-p;
                q = q/2;
                r = p;
            }
            p = p/2;
        }
    } // end parallel batcher sort
}

#endif //PARALLEL_MULTISCALE_APPROXIMATION_MULTISCALE_STRUCTURE_CUH
