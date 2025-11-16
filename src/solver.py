import numpy as np


def LDU_factorize(A):
    log = []
    n = A.shape[0]
    L = np.identity(n, float)
    D = np.identity(n, float)
    U = A.copy()  
    
    # Initial state
    log.append(("INIT", U.copy(), "Initial matrix"))

    # Forward elimination to form L and U
    for i in range(n):
        pivot = U[i][i]
        for j in range(i+1, n):
            below = U[j][i]
            factor = below/pivot
            
            # Record L update
            L[j][i] = factor
            log.append(("L", L.copy(), f"L[{j+1},{i+1}] = {factor:.4f}"))
            
            # Row operation
            U[j] = U[j] -  factor * U[i]
            
            # Save U step with explanation
            explanation = f"R{j+1} ← R{j+1} − ({factor:.4f}) R{i+1}"
            log.append(("U", U.copy(), explanation))
       
    # Diagonal normalization to extract D     
    for i in range(n):
        D[i][i] = U[i][i]
        log.append(("D", D.copy(), f"D[{i+1},{i+1}] = {U[i][i]:.4f}"))
        
        U[i] = U[i] / D[i][i]
        log.append(("U", U.copy(), f"R{i+1} ← (1 / {D[i][i]:.4f}) R{i+1}"))

    return L, D, U, log


def gaussian_elimination(A, b):
    log = []
    arg = np.column_stack((A.astype(float), b.astype(float)))
    n = arg.shape[0]
    
    # forward elimination
    for i in range(n):
        pivot = arg[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError(f"Zero or small at row {i+1}, need row exchange.")
        
        for j in range(i+1, n):
            below = arg[j][i]
            ell = below/pivot
            arg[j] = arg[j] -  ell * arg[i] 
            des = f"R{j+1} - ({ell}*R{i+1}) -> R{j+1}"
            log.append((arg.copy(), des))
            
    # back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        rhs = arg[i, -1] - np.dot(arg[i, i+1:n], x[i+1:n])
        x[i] = rhs / arg[i, i]
    
    return arg, x, log


def permutation_matrix(A):
    n = A.shape[0]
    P = np.eye(n)
    A_temp = A.copy().astype(float)
    
    for i in range(n-1):
        pivot_row = i + np.argmax(np.abs(A[i:, i]))
        if pivot_row != i:
            P[[i, pivot_row]] = P[[pivot_row, i]]
            A_temp[[i, pivot_row]] = A_temp[[pivot_row, i]]
    return P
            
