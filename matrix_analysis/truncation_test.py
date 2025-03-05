import numpy as np
import matplotlib.pyplot as plt
import sys


# when running the code above is demanding, is suggested to run the code below for L = 3, ..., 7
def collect_info(level, max_level, max_T):
    """
    function to compute the code above for a given level (max 7)
    """
    # load data
    M = np.load(f"M_{max_level}.npy")  # M largest matrix
    t_mask = np.load(f"t_mask_{max_level}.npy")  # mask for fast computation of truncated matrix
    # specific for our test settings
    cum_count = [0, 9, 34, 115, 404, 1493, 5718, 22359]
    # higher matrices are not feasible for memory reason

    # compute nnz info
    # set up list where we will store norms and nnz info for plot purposes
    nnz_ratio = []
    # compute nnz elements
    nnz = np.sum(~np.isclose(M[:cum_count[level], :cum_count[level - 1]], 0, atol=1e-8))
    # compute nnz element of the truncated version
    # indeed we count for every block column how many values are close to T*q_col
    # where, q_col is the separation distance of the set related to the column
    # q_j = 2**-(level+2) since we used equispaced points on the unit square
    for T in range(1, max_T):
        # apply the mask to get the truncated matrix
        M_t = np.where(t_mask[:cum_count[level], :cum_count[level - 1]] >= T, 0,
                       M[:cum_count[level], :cum_count[level - 1]])
        # compute the number of nnz entries
        nnz_t = np.sum(~np.isclose(M_t, 0, atol=1e-8))
        nnz_ratio.append(nnz_t / nnz)
    # computed nnz_ratios for level and T = 1, ..., 6

    # compute norm info
    # set up list where we will store norms and nnz info for plot purposes
    norm_ratio = []
    # get the normalized norm difference
    M_norm = np.linalg.norm(M[:cum_count[level], :cum_count[level - 1]], 2)
    # compute m-m(t)
    for T in range(1, max_T):
        M_diff = np.where(t_mask[:cum_count[level], :cum_count[level - 1]] < T, 0,
                          M[:cum_count[level], :cum_count[level - 1]])
        M_diff_norm = np.linalg.norm(M_diff, 2)
        norm_ratio.append(M_diff_norm / M_norm)
    # computed everything for level

    # print the results
    print(f"Level {level}:")
    print(f"||M||_2 = {M_norm:8.4f}")
    for T in range(1, max_T):
        print(f"||M-M({T})||_2/||M||_2 = {norm_ratio[T - 1]:8.4f}")
    for T in range(1, max_T):
        print(f"nnz(M({T}))/nnz(M) = {nnz_ratio[T - 1]:8.4f}")
    return norm_ratio, nnz_ratio


try:
    if len(sys.argv) == 2:
        L = int(sys.argv[1])
        trunc = 6
    else:
        L = int(sys.argv[1])
        trunc = int(sys.argv[2])
    _, axs = plt.subplots(1, 2)
    axs[0].set_title("norm")
    axs[1].set_title("nnz")
    axs[0].set_yscale("log")
    for sub_l in np.arange(2, L+1):
        norm_r, nnz_r = collect_info(sub_l, L, trunc+1)
        if norm_r[-1] > 1e-10:
            i = len(norm_r)
        else:
            i = np.where(np.array(norm_r) < 1e-10)[0][0]
        axs[0].plot(np.arange(1, i+1), norm_r[:i], label=f"level = {sub_l}")
        axs[1].plot(np.arange(1, len(nnz_r)+1), nnz_r, label=f"level = {sub_l}")
    axs[0].legend()
    axs[1].legend()
    plt.show()
except:
    print(f"unexpected behaviour with input: {sys.argv}")
    exit()
