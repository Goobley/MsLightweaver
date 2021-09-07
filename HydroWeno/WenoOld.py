import numpy as np
from numba import njit, stencil, prange

@stencil
def beta_0(V):
    return 13./12.*(V[-2] - 2.*V[-1] + V[0])**2 + 0.25*(V[-2] - 4.*V[-1] + 3.*V[0])**2

@stencil
def beta_1(V):
    return 13./12.*(V[-1] - 2.*V[0] + V[+1])**2 + 0.25*(V[-1] -V[+1])**2

@stencil
def beta_2(V):
    return 13./12.*(V[0] - 2.*V[+1] + V[+2])**2 + 0.25*(3.*V[0] - 4.*V[+1] + V[+2])**2

@njit('float64[:,:](float64[:])')
def betas(V):
    out = np.empty((3, V.shape[0]))
    beta_0(V, out=out[0, :])
    beta_1(V, out=out[1, :])
    beta_2(V, out=out[2, :])
    return out

@stencil(neighborhood=((-2,2),))
def weno_extrapolate_left(V, beta0, beta1, beta2):
    Pow = 2
    WenoEps = 1e-36

    EnoCoeff = np.array((( 15.0/8.0,  3.0/8.0, -1.0/8.0), 
                         (-10.0/8.0,  6.0/8.0,  6.0/8.0),
                         (  3.0/8.0, -1.0/8.0,  3.0/8.0)))
    LinW = np.array((5.0/16.0, 10.0/16.0, 1.0/16.0))

    nonLinW = LinW
    nonLinW[0] /= (WenoEps + beta0[0])**Pow
    nonLinW[1] /= (WenoEps + beta1[0])**Pow
    nonLinW[2] /= (WenoEps + beta2[0])**Pow
    wNorm = np.sum(nonLinW)
    nonLinW /= wNorm

    enoIntp = np.zeros(3)
    for s in range(3):
        for i in range(-2+s, 1+s):
            enoIntp[s] += V[i] * EnoCoeff[i, 2-s]

    result = np.dot(nonLinW, enoIntp)
    return result

@stencil(neighborhood=((-2,2),))
def weno_extrapolate_right(V, beta0, beta1, beta2):
    Pow = 2
    WenoEps = 1e-36

    EnoCoeff = np.array(((  3.0/8.0, -1.0/8.0,  3.0/8.0), 
                         (-10.0/8.0,  6.0/8.0,  6.0/8.0),
                         ( 15.0/8.0,  3.0/8.0, -1.0/8.0)))
    LinW = np.array((1.0/16.0, 10.0/16.0, 5.0/16.0))


    nonLinW = LinW
    nonLinW[0] /= (WenoEps + beta0[0])**Pow
    nonLinW[1] /= (WenoEps + beta1[0])**Pow
    nonLinW[2] /= (WenoEps + beta2[0])**Pow
    wNorm = np.sum(nonLinW)
    nonLinW /= wNorm

    enoIntp = np.zeros(3)
    for s in range(3):
        for i in range(-2+s, 1+s):
            enoIntp[s] += V[i] * EnoCoeff[i, s]

    result = np.dot(nonLinW, enoIntp)
    return result

@njit('float64[:,:](float64[:],float64[:,:])', parallel=False, nogil=True)
def weno_row(row, out):
    beta = betas(row)
    if out is None:
        out = np.zeros((2, row.shape[0]))
    weno_extrapolate_left(row, beta[0], beta[1], beta[2], out=out[0, :])
    weno_extrapolate_right(row, beta[0], beta[1], beta[2], out=out[1, :])
    return out

@njit('float64[:,:,:](float64[:,:])', parallel=False)
def reconstruct(q):
    result = np.zeros((q.shape[0], 2, q.shape[1]))
    for row in prange(result.shape[0]):
        weno_row(q[row], result[row])
    return result

@njit('float64[:,:,:](float64[:,:])', parallel=True, nogil=True, fastmath=True)
def reconstruct_inline(q):
    nRows, nGrid = q.shape
    result = np.empty((nRows, 2, nGrid))
    for row in range(nRows):
        beta = np.empty((3, nGrid))
        beta_0(q[row], out=beta[0, :])
        beta_1(q[row], out=beta[1, :])
        beta_2(q[row], out=beta[2, :])
        weno_extrapolate_left(q[row], beta[0], beta[1], beta[2], out=result[row, 0, :])
        weno_extrapolate_right(q[row], beta[0], beta[1], beta[2], out=result[row, 1, :])
    return result

@njit('float64[:,:,:](float64[:,:])', parallel=True)
def reconstruct_weno(q):
    nRows, nGrid = q.shape
    result = np.zeros((nRows, 2, nGrid))

    # Set up general and left/right coefficients.
    # The left/right coeffs are just flipped relative to each other
    Pow = 2
    WenoEps = 1e-36
    EnoCoeffL = np.array((( 1.875, -1.25 ,  0.375),
                          ( 0.375,  0.75 , -0.125),
                          (-0.125,  0.75 ,  0.375)))
    LinWL = np.array((5.0/16.0, 10.0/16.0, 1.0/16.0))
    EnoCoeffR = np.array((( 0.375, -1.25 ,  1.875),
                          (-0.125,  0.75 ,  0.375),
                          ( 0.375,  0.75 , -0.125)))
    LinWR = np.array((1.0/16.0, 10.0/16.0, 5.0/16.0))

    # Loop over each row in the q matrix - we parallelise over rows
    for row in prange(nRows):
        beta = np.empty(3)
        for i in range(2, nGrid-2):
            # Compute beta, the smoothness indicator for each intepolating polynomial
            beta[0] = 13./12.*(q[row, i-2] - 2.*q[row, i-1] + q[row, i])**2 + 0.25*(q[row, i-2] - 4.*q[row, i-1] + 3.*q[row, i])**2
            beta[1] = 13./12.*(q[row, i-1] - 2.*q[row, i] + q[row, i+1])**2 + 0.25*(q[row, i-1] - q[row, i+1])**2
            beta[2] = 13./12.*(q[row, i] - 2.*q[row, i+1] + q[row, i+2])**2 + 0.25*(3.*q[row, i] - 4.*q[row, i+1] + q[row, i+2])**2

            # Compute and normalise the non-linear weights
            nonLinWL = LinWL / (WenoEps + beta)**Pow
            nonLinWR = LinWR / (WenoEps + beta)**Pow
            nonLinWL /= np.sum(nonLinWL)
            nonLinWR /= np.sum(nonLinWR)

            # Compute the standard polynomial reconstructions
            enoIntpL = np.zeros(3)
            enoIntpR = np.zeros(3)
            for s in range(3):
                gridIdx = s + i - 2
                enoIntpL[s] = np.dot(q[row, gridIdx:gridIdx+3], EnoCoeffL[2-s]) 
                enoIntpR[s] = np.dot(q[row, gridIdx:gridIdx+3], EnoCoeffR[s]) 

            # Combine the different polynomial reconstrucitions weighted by their non-linear weights
            result[row, 0, i] = np.dot(nonLinWL, enoIntpL)
            result[row, 1, i] = np.dot(nonLinWR, enoIntpR)
    return result