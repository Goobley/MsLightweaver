import numpy as np
from numba import njit, stencil, prange

@njit(parallel=True, cache=True)
def reconstruct_weno(q, dx=None):
    nRows, nGrid = q.shape
    result = np.zeros((nRows, 2, nGrid))

    # Set up general and left/right coefficients.
    # The left/right coeffs are just flipped relative to each other
    Pow = 2
    WenoEps = 1e-6
    EnoCoeffL = np.array((( 11.0/6.0, -7.0/6.0,  2.0/6.0),
                          ( 2.0/6.0,   5.0/6.0, -1.0/6.0),
                          (-1.0/6.0,   5.0/6.0,  2.0/6.0)))
    LinWL = np.array([0.3, 0.6, 0.1])
    EnoCoeffR = np.array((( 2.0/6.0,  -7.0/6.0,  11.0/6.0),
                          (-1.0/6.0,   5.0/6.0,  2.0/6.0),
                          ( 2.0/6.0,   5.0/6.0, -1.0/6.0)))
    LinWR = np.array([0.1, 0.6, 0.3])

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
    result[:, 0, :2] = q[:, :2]
    result[:, 1, :2] = q[:, :2]
    result[:, 0, -2:] = q[:, -2:]
    result[:, 1, -2:] = q[:, -2:]
    return result

@njit(parallel=True, cache=True)
def reconstruct_weno_log(q, dx=None):
    nRows, nGrid = q.shape
    result = np.zeros((nRows, 2, nGrid))

    # Set up general and left/right coefficients.
    # The left/right coeffs are just flipped relative to each other
    Pow = 2
    WenoEps = 1e-6
    EnoCoeffL = np.array((( 11.0/6.0, -7.0/6.0,  2.0/6.0),
                          ( 2.0/6.0,   5.0/6.0, -1.0/6.0),
                          (-1.0/6.0,   5.0/6.0,  2.0/6.0)))
    LinWL = np.array([0.3, 0.6, 0.1])
    EnoCoeffR = np.array((( 2.0/6.0,  -7.0/6.0,  11.0/6.0),
                          (-1.0/6.0,   5.0/6.0,  2.0/6.0),
                          ( 2.0/6.0,   5.0/6.0, -1.0/6.0)))
    LinWR = np.array([0.1, 0.6, 0.3])

    q[0] = np.log(q[0])
    q[2] = np.log(q[2])
    q[3] = np.log(q[3])

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
    result[:, 0, :2] = q[:, :2]
    result[:, 1, :2] = q[:, :2]
    result[:, 0, -2:] = q[:, -2:]
    result[:, 1, -2:] = q[:, -2:]

    result[0] = np.exp(result[0])
    result[2] = np.exp(result[2])
    result[3] = np.exp(result[3])
    return result

@njit(parallel=True, cache=True)
def reconstruct_weno_z(q, dx=None):
    nRows, nGrid = q.shape
    result = np.zeros((nRows, 2, nGrid))

    # Set up general and left/right coefficients.
    # The left/right coeffs are just flipped relative to each other
    Pow = 2
    WenoEps = 1e-10
    EnoCoeffL = np.array((( 11.0/6.0, -7.0/6.0,  2.0/6.0),
                          ( 2.0/6.0,   5.0/6.0, -1.0/6.0),
                          (-1.0/6.0,   5.0/6.0,  2.0/6.0)))
    LinWL = np.array([0.3, 0.6, 0.1])
    EnoCoeffR = np.array((( 2.0/6.0,  -7.0/6.0,  11.0/6.0),
                          (-1.0/6.0,   5.0/6.0,  2.0/6.0),
                          ( 2.0/6.0,   5.0/6.0, -1.0/6.0)))
    LinWR = np.array([0.1, 0.6, 0.3])

    # Loop over each row in the q matrix - we parallelise over rows
    for row in prange(nRows):
        beta = np.empty(3)
        betaZ = np.empty(3)
        for i in range(2, nGrid-2):
            # Compute beta, the smoothness indicator for each intepolating polynomial
            beta[0] = 13./12.*(q[row, i-2] - 2.*q[row, i-1] + q[row, i])**2 + 0.25*(q[row, i-2] - 4.*q[row, i-1] + 3.*q[row, i])**2
            beta[1] = 13./12.*(q[row, i-1] - 2.*q[row, i] + q[row, i+1])**2 + 0.25*(q[row, i-1] - q[row, i+1])**2
            beta[2] = 13./12.*(q[row, i] - 2.*q[row, i+1] + q[row, i+2])**2 + 0.25*(3.*q[row, i] - 4.*q[row, i+1] + q[row, i+2])**2
            tau5 = np.abs(beta[0] - beta[2])
            betaZ[:] = ((beta + WenoEps) / (beta + tau5 + WenoEps))

            # Compute and normalise the non-linear weights
            nonLinWL = LinWL / betaZ
            nonLinWR = LinWR / betaZ
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
    result[:, 0, :2] = q[:, :2]
    result[:, 1, :2] = q[:, :2]
    result[:, 0, -2:] = q[:, -2:]
    result[:, 1, -2:] = q[:, -2:]
    return result

@njit(parallel=True, cache=True)
def reconstruct_weno_nm(q, dx):
    nRows, nGrid = q.shape
    result = np.zeros((nRows, 2, nGrid))

    # Set up general coefficients.
    Pow = 2
    WenoEps = 1e-34
    EnoCoeff  = np.array((( 2.0,      -1.0/3.0, -2.0/3.0),
                          (-1.0/3.0,   2.0/3.0,  2.0/3.0),
                          ( 2.0/3.0,   2.0/3.0, -1.0/3.0)))
    BetaCoeffPart1 = np.array(((0.0,  2.0, -2.0),
                               (2.0, -4.0,  2.0),
                               (2.0, -4.0,  2.0)))
    BetaCoeffPart2 = np.array((( 4.0, -2.0, -2.0),
                               (-2.0,  0.0,  2.0),
                               (-6.0,  8.0, -2.0)))
    LinW = np.array([0.1, 0.6, 0.3])
    # length ratio designated rho
    rho = np.zeros(nGrid)
    rho[:-1] = dx[:-1] / (dx[:-1] + dx[1:])

    # Loop over each row in the q matrix - we parallelise over rows
    for row in prange(nRows):
        betaL = np.empty(3)
        betaR = np.empty(3)
        stencilL = np.empty((3, 3)) # coords are [stencilIdx, stencilEntry]
        stencilR = np.empty((3, 3)) # coords are [stencilIdx, stencilEntry]
        enoIntpL = np.empty(3)
        enoIntpR = np.empty(3)
        for i in range(2, nGrid-2):
            # Compute beta, the smoothness indicator for each intepolating polynomial

            q00 = q[row, i-1] + rho[i-1] * (q[row, i-1] - q[row, i-2])
            q01 = (1.0 - rho[i]) * q[row, i] + rho[i] * q[row, i+1]
            q10 = (1.0 - rho[i-1]) * q[row, i-1] + rho[i-1] * q[row, i]
            q11 = q[row, i+1] + rho[i+1] * (q[row, i+1] - q[row, i+2])
            q12 = (1.0 - rho[i+1]) * q[row, i+1] + rho[i+1] * q[row, i+2]
            q21 = (1.0 - rho[i-2]) * q[row, i-2] + rho[i-2] * q[row, i-1]
            
            stencilL[0, 0] = q[row, i]
            stencilL[0, 1] = q01
            stencilL[0, 2] = q11
            stencilL[1, 0] = q01
            stencilL[1, 1] = q[row, i]
            stencilL[1, 2] = q10
            stencilL[2, 0] = q10
            stencilL[2, 1] = q[row, i-1]
            stencilL[2, 2] = q21
            
            stencilR[0, 0] = q[row, i]
            stencilR[0, 1] = q10
            stencilR[0, 2] = q00
            stencilR[1, 0] = q10
            stencilR[1, 1] = q[row, i]
            stencilR[1, 2] = q01
            stencilR[2, 0] = q01
            stencilR[2, 1] = q[row, i+1]
            stencilR[2, 2] = q12


            for s in range(3):
                betaL[s] = 13.0/12.0*np.dot(BetaCoeffPart1[s], stencilL[s])**2 + 0.25*np.dot(BetaCoeffPart2[s], stencilL[s])**2
                betaR[s] = 13.0/12.0*np.dot(BetaCoeffPart1[s], stencilR[s])**2 + 0.25*np.dot(BetaCoeffPart2[s], stencilR[s])**2

            # Compute and normalise the non-linear weights
            nonLinWL = LinW / (WenoEps + betaL)**Pow
            nonLinWR = LinW / (WenoEps + betaR)**Pow
            nonLinWL /= np.sum(nonLinWL)
            nonLinWR /= np.sum(nonLinWR)

            # Compute the standard polynomial reconstructions
            for s in range(3):
                enoIntpL[s] = np.dot(stencilL[s], EnoCoeff[s])
                enoIntpR[s] = np.dot(stencilR[s], EnoCoeff[s])

            # Combine the different polynomial reconstrucitions weighted by their non-linear weights
            result[row, 0, i] = np.dot(nonLinWL, enoIntpL)
            result[row, 1, i] = np.dot(nonLinWR, enoIntpR)
    result[:, 0, :2] = q[:, :2]
    result[:, 1, :2] = q[:, :2]
    result[:, 0, -2:] = q[:, -2:]
    result[:, 1, -2:] = q[:, -2:]
    return result

@njit(parallel=True, cache=True)
def reconstruct_weno_nm_z(q, dx):
    nRows, nGrid = q.shape
    result = np.zeros((nRows, 2, nGrid))

    # Set up general coefficients.
    Pow = 2
    WenoEps = 1e-34
    EnoCoeff  = np.array((( 2.0,      -1.0/3.0, -2.0/3.0),
                          (-1.0/3.0,   2.0/3.0,  2.0/3.0),
                          ( 2.0/3.0,   2.0/3.0, -1.0/3.0)))
    BetaCoeffPart1 = np.array(((0.0,  2.0, -2.0),
                               (2.0, -4.0,  2.0),
                               (2.0, -4.0,  2.0)))
    BetaCoeffPart2 = np.array((( 4.0, -2.0, -2.0),
                               (-2.0,  0.0,  2.0),
                               (-6.0,  8.0, -2.0)))
    LinW = np.array([0.1, 0.6, 0.3])
    # length ratio designated rho
    rho = np.zeros(nGrid)
    rho[:-1] = dx[:-1] / (dx[:-1] + dx[1:])

    # Loop over each row in the q matrix - we parallelise over rows
    for row in prange(nRows):
        betaL = np.empty(3)
        betaR = np.empty(3)
        betaZL = np.empty(3)
        betaZR = np.empty(3)
        stencilL = np.empty((3, 3)) # coords are [stencilIdx, stencilEntry]
        stencilR = np.empty((3, 3)) # coords are [stencilIdx, stencilEntry]
        enoIntpL = np.empty(3)
        enoIntpR = np.empty(3)
        for i in range(2, nGrid-2):
            # Compute beta, the smoothness indicator for each intepolating polynomial

            q00 = q[row, i-1] + rho[i-1] * (q[row, i-1] - q[row, i-2])
            q01 = (1.0 - rho[i]) * q[row, i] + rho[i] * q[row, i+1]
            q10 = (1.0 - rho[i-1]) * q[row, i-1] + rho[i-1] * q[row, i]
            q11 = q[row, i+1] + rho[i+1] * (q[row, i+1] - q[row, i+2])
            q12 = (1.0 - rho[i+1]) * q[row, i+1] + rho[i+1] * q[row, i+2]
            q21 = (1.0 - rho[i-2]) * q[row, i-2] + rho[i-2] * q[row, i-1]
            
            stencilL[0, 0] = q[row, i]
            stencilL[0, 1] = q01
            stencilL[0, 2] = q11
            stencilL[1, 0] = q01
            stencilL[1, 1] = q[row, i]
            stencilL[1, 2] = q10
            stencilL[2, 0] = q10
            stencilL[2, 1] = q[row, i-1]
            stencilL[2, 2] = q21
            
            stencilR[0, 0] = q[row, i]
            stencilR[0, 1] = q10
            stencilR[0, 2] = q00
            stencilR[1, 0] = q10
            stencilR[1, 1] = q[row, i]
            stencilR[1, 2] = q01
            stencilR[2, 0] = q01
            stencilR[2, 1] = q[row, i+1]
            stencilR[2, 2] = q12


            for s in range(3):
                betaL[s] = 13.0/12.0*np.dot(BetaCoeffPart1[s], stencilL[s])**2 + 0.25*np.dot(BetaCoeffPart2[s], stencilL[s])**2
                betaR[s] = 13.0/12.0*np.dot(BetaCoeffPart1[s], stencilR[s])**2 + 0.25*np.dot(BetaCoeffPart2[s], stencilR[s])**2

            tau5L = np.abs(betaL[0] - betaL[2])
            tau5R = np.abs(betaR[0] - betaR[2])
            betaZL[:] = ((betaL + WenoEps) / (betaL + tau5L + WenoEps))
            betaZR[:] = ((betaR + WenoEps) / (betaR + tau5R + WenoEps))

            # Compute and normalise the non-linear weights
            nonLinWL = LinW / betaZL
            nonLinWR = LinW / betaZR
            nonLinWL /= np.sum(nonLinWL)
            nonLinWR /= np.sum(nonLinWR)

            # Compute the standard polynomial reconstructions
            for s in range(3):
                enoIntpL[s] = np.dot(stencilL[s], EnoCoeff[s])
                enoIntpR[s] = np.dot(stencilR[s], EnoCoeff[s])

            # Combine the different polynomial reconstrucitions weighted by their non-linear weights
            result[row, 0, i] = np.dot(nonLinWL, enoIntpL)
            result[row, 1, i] = np.dot(nonLinWR, enoIntpR)
    result[:, 0, :2] = q[:, :2]
    result[:, 1, :2] = q[:, :2]
    result[:, 0, -2:] = q[:, -2:]
    result[:, 1, -2:] = q[:, -2:]
    return result