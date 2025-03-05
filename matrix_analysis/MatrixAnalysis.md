## Truncation matrix analysis

The primary focus of this analysis is to investigate the computational gain, linked to the sparsity of the truncation process, and the associated loss of information.

The experiments are carried with python (3.8.10) an numpy (1.24.4).

From the files directory issue

```
python3 truncation_pre.py
```

to build and store the matrix $M_{6}$ as `M_6.npy` and mask matrix which will be exploited to compute truncations for different values of $T$ as `t_mask_6.npy`. Then, 

```
python3 truncation_test.py 6

```

will produce the desired results. 

Results for $L=7$ can be achieved building the $L=6$ matrix and then appendind the $L=7$ row. Our workstation cannot handle the direct construction. 
