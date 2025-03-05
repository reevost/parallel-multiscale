import numpy as np
import sys


# define function to compute points at level "step"
def equispaced_on_square(step):
    x = np.linspace(0, 1, 2**step+1)
    y = np.linspace(0, 1, 2**step+1)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


# set up the matrix and the truncated matrix
# auxiliary functions
def dist_matrix_2d(set1, set2):
    N = len(set1)
    M = len(set2)
    d_matrix = np.empty((N, M), dtype=np.float64)
    for col in range(M):
        d_matrix[:, col] = np.sum((set1-set2[col])**2, axis=1)**.5
    return d_matrix


# C2 wendland (3d) (1-x)_{+}(4x+1)
def phi_matrix(set1, set2, delta=1):
    dm = dist_matrix_2d(set1, set2)/delta
    return (np.where(1-dm < 0, 0, 1-dm)**4 * (4*dm + 1))/delta**2


try:
    if len(sys.argv) == 2:
        L = int(sys.argv[1])
    else:
        L = 6  # more than 7 means matrices too large to be handled this naively
    # we store the cumulative number of points to have access to the single levels
    # here cum_count[level+1] is the cumulative number of points at level "level"
    # cum_count[level+1]-cum_count[level] are the points at level "level"
    cum_count = [0]
    # cumulative set of points construction (associated to the M matrix)
    cumulative_points = equispaced_on_square(1)
    cum_count += [len(cumulative_points)]
    for level in range(2, L + 1):
        tp = equispaced_on_square(level)
        cumulative_points = np.row_stack([cumulative_points, tp])
        cum_count += [len(cumulative_points)]
    M = np.zeros((cum_count[-1], cum_count[-2]))
    for level in range(L-1):
        # set of point in the level
        X_l = cumulative_points[cum_count[level]: cum_count[level+1], :]
        # delta_l = nu * h_l (chosen nu = 4 for all experiments)
        # h_l in ou settings is sqrt(2)/2**(l+1) for l=1, ..., L
        # delta_l = 4 * np.sqrt(2) / 2**(level+2) == np.sqrt(2)/2**level
        delta_l = np.sqrt(2)/2**level
        A_l = phi_matrix(X_l, X_l, delta_l)
        # inversion of the matrix
        U, S, Vt = np.linalg.svd(A_l)
        # the inverse is Vt.T @ np.diag(1/S) @ U.T
        # build the block column B
        B = phi_matrix(cumulative_points[cum_count[level+1]:], X_l, delta_l)
        # compute B A^{-1} and store the result on M
        M[cum_count[level+1]:, cum_count[level]: cum_count[level+1]] = B @ Vt.T @ np.diag(1/S) @ U.T
        print(f"done {level}")

    # set up a matrix with entries that tells for which value of "T" the entry is kept in the truncation
    truncation_mask = np.empty(M.shape, dtype=np.int8)
    for level in range(L-1):
        X_l = cumulative_points[cum_count[level]: cum_count[level + 1], :]
        # in our settings the separation distance
        q_l = 2**-(level+2)
        dist_mat = dist_matrix_2d(cumulative_points[cum_count[level+1]:, :], X_l)/q_l
        truncation_mask[cum_count[level + 1]:, cum_count[level]: cum_count[level + 1]] = np.floor(dist_mat).astype(np.int8)

    np.save(f"M_{L}", M)
    np.save(f"t_mask_{L}", truncation_mask)
except:
    print(f"unexpected behaviour with input: {sys.argv}")
    exit()