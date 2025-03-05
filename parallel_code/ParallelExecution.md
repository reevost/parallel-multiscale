## Hardware requirements

The numerical examples have been performed on an *NVIDIA A100* GPU of the the [Marc3a](https://www.hkhlr.de/en/clusters/marc3a-cluster-marburg) cluster.

## Software requirements

We employed `bash` as shell and the `slurm` scheduler was used to having access to the resources.

The modules needed are the following:

```
cmake 3.21.3
cuda 11.1
openblas 0.3.7
```

In our setting we needed *gnu9* to load *openblas*. Other *blas* implementation should lead to the same results but times may differs.

## Compilation and execution

To compile the code, first reach the folder with all the files then run

```
mkdir 2D_domain
mkdir logs
cmake -B build
cmake --build build
```

Then a job to compute the results can be submitted
```
sbatch multiscale_approximation_job.sh
```

In this way all times for the parallel execution of the multiscale approximation are stored locally in *multiscale_times_LX* for X = 7, ..., 11. 
To compute the efficiency we used

$$
E = \frac{T_{serial}}{p T_{p}},
$$

where $T_{p}$ is the time obtained with $p$ processors (recall that $p$ = #warps $\times 32$) and $T_{serial}$ is obtained by manually replacing the execution configuration in `iterative_parallel_methods` with `<<<1, 1>>>`, update line 110 and 131 of `main_times.cu` with respectively
```
size_t filePathLength = snprintf(file_path_domain, 200, "/%dD_domain/multiscale_scalar_times_L%d.txt", POINTS_DIM, number_of_levels);
```
```
for (int given_warps = 1; given_warps < 2; given_warps *= 2){
```
In this way we perform the computation just once and we wont overwrite previous results. Take into account that the waiting times increase significantly. Indeed, the third line in `multiscale_approximation_job.sh` should be updated with
`#SBATCH --time=3-00:00:00` for L = 7, ..., 10 and `#SBATCH --time=7-00:00:00` for L = 11.

## Additional information

When there is interest on the solution of the scheme presented on the paper is suffient to uncomment lines in `CMakeLists.txt` and `multiscale_approximation_job.sh`.
Indeed, in `./2D_domain/results_in_121_points`, for every level can be found the approximantion, the error and the approximant up to that level.

For accurate evaluation, the macro *EVALUATION_POINTS_ON_AXIS* can be modified (this might change the storing directory, be sure to create one) and the *EPS* macro control the tolerance for the conjugate gradient routine. Both can be found in `macro.h`

Additionally the parameters *mu* and *nu* can be changed during the execution adding respectively a second and third paramenter to the executable *parallel_multiscale_times* or the third and fourth parameters in the executables *parallel_multiscale*.
The second paramenter of *parallel_multiscale* is the number warps involved in the approximation process.
