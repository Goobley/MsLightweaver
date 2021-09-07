import numpy as np
from numba import njit

# @njit(cache=True)
def mass_flux(recon):
    result = np.empty_like(recon)
    result[0] = 0.0
    result[1:] = recon[0] * recon[1:]
    return result

# @njit(cache=True)
def advection_flux(recon):
    fluxLR = np.zeros_like(recon)
    fluxLR[:, 0, :] = mass_flux(recon[:, 0, :])
    fluxLR[:, 1, :] = mass_flux(recon[:, 1, :])
    maxC = np.zeros(recon.shape[-1])
    maxC[1:] = 0.5 * (np.abs(recon[0, 1, :-1]) + np.abs(recon[0, 0, 1:]))
    flux = np.zeros((recon.shape[0], recon.shape[-1]))
    flux[:, 1:] = 0.5 * (fluxLR[:, 1, :-1] + fluxLR[:, 0, 1:] - maxC[1:] * (recon[:, 0, 1:] - recon[:, 1, :-1]))
    return flux

class Advector:
    def __init__(self, grid, data, apply_bcs, reconstruct):
        self.grid = grid
        self.data = data
        self.apply_bcs = apply_bcs
        self.reconstruct = reconstruct

    def step(self, dt):
        GriBeg = self.grid.griBeg
        GriEnd = self.grid.griEnd
        dtx = (dt / self.grid.dx)[GriBeg:GriEnd]

        self.apply_bcs(self.grid, self.data)
        recon = self.reconstruct(self.data, self.grid.dx)
        flux = advection_flux(recon)
        newData = np.copy(self.data)
        newData[:, GriBeg:GriEnd] = self.data[:, GriBeg:GriEnd] - dtx * (flux[:, GriBeg+1:GriEnd+1] - flux[:, GriBeg:GriEnd])

        self.apply_bcs(self.grid, newData)
        recon = self.reconstruct(newData, self.grid.dx)
        flux = advection_flux(recon)
        newData2 = np.copy(self.data)
        newData2[:, GriBeg:GriEnd] = 0.75 * self.data[:, GriBeg:GriEnd] + 0.25 * newData[:, GriBeg:GriEnd] - 0.25 * dtx * (flux[:, GriBeg+1:GriEnd+1] - flux[:, GriBeg:GriEnd])

        self.apply_bcs(self.grid, newData2)
        recon = self.reconstruct(newData2, self.grid.dx)
        flux = advection_flux(recon)
        newData3 = np.copy(self.data)
        newData3[:, GriBeg:GriEnd] = 1/3 * self.data[:, GriBeg:GriEnd] + 2/3 * newData2[:, GriBeg:GriEnd] - 2/3 * dtx * (flux[:, GriBeg+1:GriEnd+1] - flux[:, GriBeg:GriEnd])

        self.data = newData3


