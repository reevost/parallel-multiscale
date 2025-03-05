#ifndef PARALLEL_MULTISCALE_APPROXIMATION_MACROS_H
#define PARALLEL_MULTISCALE_APPROXIMATION_MACROS_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

#define POINTS_DIM 2
#define VALUES_DIM 1
#define EPS 0.0001
#define EVALUATION_POINTS_ON_AXIS 11

#endif //PARALLEL_MULTISCALE_APPROXIMATION_MACROS_H
