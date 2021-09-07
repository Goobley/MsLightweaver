import numpy as np
from .Weno import reconstruct_weno, reconstruct_weno_z, reconstruct_weno_nm
from .Flux import lax_friedrichs_flux, lax_wendroff_flux, gforce_flux, lax_friedrichs_flux_cons
from .PrimCons import cons2prim, prim2cons
from .SimSettings import Prim, Sim

SmallDens = 1e-30
SmallVelo = -np.inf
SmallPres = 1e-30
SmallEint = 1e-30
PrimLimits = np.array([SmallDens, SmallVelo, SmallPres, SmallEint])
def limit_recon(primRecon):
    primRecon[:] = np.fmax(primRecon, PrimLimits[:, None, None])
    return primRecon

def tvd_rk3(apply_bcs, reconstruct, stencilWidth, flux_fn, reconstructionValidator):
    def fix_reconstruction(primRecon, prim, grid):
        if reconstructionValidator is not None:
           okay, idxs = reconstructionValidator.validate(primRecon, grid)
           if not okay:
               start = max(idxs[0] - stencilWidth, 0)
               end = min(idxs[-1] + stencilWidth + 1, grid.griMax)
               trans = reconstructionValidator.transform(prim[:, start:end])
               reconTrans = reconstruct(trans, grid.dx[start:end])
               recon = reconstructionValidator.inverse_transform(reconTrans)
               primRecon[:, :, start+stencilWidth//2:end-stencilWidth//2] = recon[:, :, stencilWidth//2:-(stencilWidth//2)]

    def tvd_rk3_impl(dt, grid, prim, cons):
        # dx = np.ones(prim.shape[1]) * Sim.GrDx
        consNew = prim2cons(prim)
        GriBeg = grid.griBeg
        GriEnd = grid.griEnd
        dtx = (dt / grid.dx)[GriBeg:GriEnd]

        apply_bcs(grid, prim)
        primRecon = reconstruct(prim, grid.dx)
        fix_reconstruction(primRecon, prim, grid)
        flux = flux_fn(primRecon, grid.dx, dt)
        consNew[:, GriBeg:GriEnd] = cons[:, GriBeg:GriEnd] - dtx * (flux[:, GriBeg+1:GriEnd+1] - flux[:, GriBeg:GriEnd])
        primNew = cons2prim(consNew)

        apply_bcs(grid, primNew)
        primRecon[:] = reconstruct(primNew, grid.dx)
        fix_reconstruction(primRecon, primNew, grid)
        flux[:] = flux_fn(primRecon, grid.dx, dt)
        consNew[:, GriBeg:GriEnd] = 0.75 * cons[:, GriBeg:GriEnd] + 0.25 * consNew[:, GriBeg:GriEnd] - 0.25 * dtx * (flux[:, GriBeg+1:GriEnd+1] - flux[:, GriBeg:GriEnd]) 
        primNew[:] = cons2prim(consNew)

        apply_bcs(grid, primNew)
        primRecon[:] = reconstruct(primNew, grid.dx)
        fix_reconstruction(primRecon, primNew, grid)
        flux[:] = flux_fn(primRecon, grid.dx, dt)
        consNew[:, GriBeg:GriEnd] = 1/3 * cons[:, GriBeg:GriEnd] + 2/3 * consNew[:, GriBeg:GriEnd] - 2/3 * dtx * (flux[:, GriBeg+1:GriEnd+1] - flux[:, GriBeg:GriEnd]) 
        primNew[:] = cons2prim(consNew)

        return primNew, consNew
    return tvd_rk3_impl


def tvd_rk3_cons(apply_bcs, reconstruct, flux_fn):
    def tvd_rk3_impl(dt, grid, prim, cons):
        # dx = np.ones(prim.shape[1]) * Sim.GrDx
        consNew = np.zeros_like(cons)
        GriBeg = grid.griBeg
        GriEnd = grid.griEnd
        dtx = (dt / grid.dx)[GriBeg:GriEnd]

        apply_bcs(grid, prim)
        consNew[:] = prim2cons(prim)
        consRecon = reconstruct(cons, grid.dx)
        # primRecon[:] = limit_recon(primRecon)
        flux = flux_fn(consRecon, grid.dx, dt)
        consNew[:, GriBeg:GriEnd] = cons[:, GriBeg:GriEnd] - dtx * (flux[:, GriBeg+1:GriEnd+1] - flux[:, GriBeg:GriEnd])
        primNew = cons2prim(consNew)

        apply_bcs(grid, primNew)
        consNew[:] = prim2cons(primNew)
        consRecon[:] = reconstruct(consNew, grid.dx)
        # primRecon[:] = limit_recon(primRecon)
        flux[:] = flux_fn(consRecon, grid.dx, dt)
        consNew[:, GriBeg:GriEnd] = 0.75 * cons[:, GriBeg:GriEnd] + 0.25 * consNew[:, GriBeg:GriEnd] - 0.25 * dtx * (flux[:, GriBeg+1:GriEnd+1] - flux[:, GriBeg:GriEnd]) 
        primNew[:] = cons2prim(consNew)

        apply_bcs(grid, primNew)
        consNew[:] = prim2cons(primNew)
        consRecon[:] = reconstruct(consNew, grid.dx)
        # primRecon[:] = limit_recon(primRecon)
        flux[:] = flux_fn(consRecon, grid.dx, dt)
        consNew[:, GriBeg:GriEnd] = 1/3 * cons[:, GriBeg:GriEnd] + 2/3 * consNew[:, GriBeg:GriEnd] - 2/3 * dtx * (flux[:, GriBeg+1:GriEnd+1] - flux[:, GriBeg:GriEnd]) 
        primNew[:] = cons2prim(consNew)

        return primNew, consNew
    return tvd_rk3_impl
