import numpy as np

def cost(X, solution, r, n, m, weights):
    cost0 = 0
    for j in range(m):
        aux = 0
        for i in range(n):
            aux += (X[j, i] - solution[i]) ** 2
        aux -= r[j] ** 2
        cost0 += weights[j] * (aux ** 2)
    return cost0
    
def trilateration_opt(X0, r):
    weights = np.ones(len(r))
    n = X0.shape[1]
    m = X0.shape[0]

    # Using X0 as a matrix directly instead of flattening it
    X = np.copy(X0)
    A = np.zeros((n, n))
    b = np.zeros(n)
    translation = np.zeros(n)
    aux2 = np.zeros((n, n))
    I_n = np.zeros((n, n))

    # Normalize weights
    weights_sum = sum(weights)
    weights = [w / weights_sum for w in weights]

    # Apply translation
    for i in range(n):
        sum_weights = sum(weights[j] * X[j, i] for j in range(m))
        translation[i] = sum_weights
        for j in range(m):
            X[j, i] -= translation[i]

    # Compute A
    val_diag = np.zeros(m)
    for j in range(m):
        sum_sq = sum(X[j, i] ** 2 for i in range(n))
        val_diag[j] = weights[j] * (sum_sq - r[j] ** 2)
        for i in range(n):
            I_n[i, i] += val_diag[j]

        m1 = X[j].reshape(n, 1)
        aux1 = 2 * weights[j] * np.dot(m1, m1.T)
        aux2 += aux1

    A = I_n + aux2

    # Compute b
    for j in range(m):
        val_diag[j] *= -1
        for i in range(n):
            b[i] += val_diag[j] * X[j, i]

    # Compute Eigenvalues and Eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(A)
    U = eig_vecs.real
    U_T = U.T
    D = np.diag(eig_vals.real)

    # Change of variable
    b = np.dot(U_T, b)

    # Build M matrix
    dim_M = 2 * n + 1
    M = np.zeros((dim_M, dim_M))
    for i in range(n):
        M[i, i + n] = -b[i]
        M[i + n, i + n] = -D[i, i]
        M[i, i] = -D[i, i]
        M[i + n, 2 * n] = -b[i]
        M[2 * n, i] = 1

    # Eigenvalue decomposition of M
    eig_vals_M, eig_vecs_M = np.linalg.eig(M)
    U_M_re = eig_vecs_M.real

    for j in range(2 * n + 1):
        U_M_re[:, j] /= U_M_re[2 * n, j]

    solution_opt = np.zeros(n)
    cost_min = float('inf')

    for j in range(2 * n + 1):
        solution = U_M_re[n:, j][:n]  # Adjust solution dimension
        solution = np.dot(U, solution) + translation

        cost0 = cost(X0, solution, r, n, m, weights)
        if cost0 < cost_min:
            cost_min = cost0
            solution_opt = solution.copy()

    return solution_opt