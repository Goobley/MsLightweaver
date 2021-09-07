import numpy as np

class Grid:
    def __init__(self, cellInterfaces, numGhost):
        self.numCells = cellInterfaces.shape[0] - 1
        self.numGhost = numGhost
        leftDx = cellInterfaces[1] - cellInterfaces[0]
        rightDx = cellInterfaces[-1] - cellInterfaces[-2]
        self.interfaces = np.concatenate((np.linspace(cellInterfaces[0] - leftDx * numGhost, cellInterfaces[0] - leftDx, numGhost),
                                          cellInterfaces,
                                          np.linspace(cellInterfaces[-1] + rightDx, cellInterfaces[-1] + rightDx * numGhost, numGhost)))
        self.dx = self.interfaces[1:] - self.interfaces[:-1]
        self.cc = 0.5*(self.interfaces[1:] + self.interfaces[:-1])
        self.griBeg = self.numGhost
        self.griEnd = self.numCells + self.numGhost
        self.griMax = self.numCells + 2 * self.numGhost
        self.xStart = self.cc[self.griBeg]
        self.xEnd = self.cc[self.griEnd - 1]

class GridCC:
    def __init__(self, cellCentres, numGhost):
        self.numCells = cellCentres.shape[0]
        self.numGhost = numGhost
        leftDx = cellCentres[1] - cellCentres[0]
        rightDx = cellCentres[-1] - cellCentres[-2]
        self.cc = np.concatenate((np.linspace(cellCentres[0] - leftDx * numGhost, cellCentres[0] - leftDx, numGhost),
                                          cellCentres,
                                          np.linspace(cellCentres[-1] + rightDx, cellCentres[-1] + rightDx * numGhost, numGhost)))
        self.interfaces = np.concatenate(([self.cc[0] - leftDx], 0.5*(self.cc[1:] + self.cc[:-1]), [self.cc[-1] + rightDx]))
        self.dx = self.interfaces[1:] - self.interfaces[:-1]
        self.griBeg = self.numGhost
        self.griEnd = self.numCells + self.numGhost
        self.griMax = self.numCells + 2 * self.numGhost
        self.xStart = self.cc[self.griBeg]
        self.xEnd = self.cc[self.griEnd - 1]

class Simulation:
    def __init__(self, grid, initialise, apply_bcs, timestep_limit, time_integrate, reconstruction, stencilWidth, flux_fn, reconstructionValidator=None):
        self.grid = grid
        self.timestep_limit  = timestep_limit
        self.V, self.U = initialise(grid)
        self.time_integrate = time_integrate(apply_bcs, reconstruction, stencilWidth, flux_fn, reconstructionValidator)
        self.apply_bcs = apply_bcs

    def step(self, tMax=None):
        dt = self.timestep_limit(self.grid, self.V)
        if tMax is not None:
            dt = min(dt, tMax)
        self.V[:], self.U[:] = self.time_integrate(dt, self.grid, self.V, self.U)
        self.apply_bcs(self.grid, self.V)
        return dt



