import pickle
import numpy as np
from lightweaver.atmosphere import Atmosphere, ScaleType
from lightweaver.atomic_table import DefaultAtomicAbundance
from lightweaver.atomic_set import RadiativeSet
from lightweaver.molecule import MolecularTable
from lightweaver.utils import InitialSolution, ConvergenceError, compute_radiative_losses, integrate_line_losses
import lightweaver.constants as Const
import lightweaver as lw
from radynpy.matsplotlib import OpcFile
from radynpy.utils import hydrogen_absorption
from numba import njit
from weno4 import weno4
from RadynAdvection import an_sol, an_rad_sol, an_gml_sol

def weno4_pos(xs, xp, fp, **kwargs):
    return np.exp(weno4(xs, xp, np.log(fp), **kwargs))

# https://stackoverflow.com/a/21901260
import subprocess
def mslightweaver_revision():
    p = Path(__file__).parent
    isGitRepo = subprocess.check_output(['git', 'rev-parse', '--is-inside-work-tree'], cwd=p).decode('ascii').strip() == 'true'
    if not isGitRepo:
        raise ValueError('Cannot find git info.')

    gitChanges = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=p).decode('ascii').strip()
    if len(gitChanges) > 0:
        raise ValueError('Uncommitted changes to tracked files, cannot procede:\n%s' % gitChanges)

    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=p).decode('ascii').strip()

def check_write_git_revision(outputDir):
    revision = mslightweaver_revision()
    with open(outputDir + 'GitRevision.txt', 'w') as f:
        f.write(revision)

def nr_advect(atmost, i0, eqPops, activeAtomNames, abundances):
    d1 = atmost.d1[i0+1]
    for a in activeAtomNames:
        pop = np.zeros_like(eqPops[a])
        for i in range(pop.shape[0]):
            pop[i, :] = an_sol(atmost, i0, eqPops[a][i], tol=1e-8, maxIter=1000)
        nTotal = d1 / (abundances.massPerH * lw.Amu) * abundances[a]
        popCorrectionFactor = nTotal / pop.sum(axis=0)
        print('Max Correction %s: %.2e' % (a, np.abs(1-popCorrectionFactor).max()))
        pop *= popCorrectionFactor
        eqPops[a][...] = pop

class CoronalIrraditation(lw.BoundaryCondition):
    def __init__(self):
        # NOTE(cmo): This data needs to be in (mu, toObs) order, i.e. mu[0]
        # down, mu[0] up, mu[1] down...
        # self.I = I1d.reshape(I1d.shape[0], -1, I1d.shape[-1])
        self.I = None

    def set_bc(self, I1d):
        self.I = np.expand_dims(I1d, axis=2)

    def compute_bc(self, atmos, spect):
        # if spect.wavelength.shape[0] != self.I.shape[0]:
        #     result = np.ones((spect.wavelength.shape[0], spect.I.shape[1], atmos.Nz))
        # else:
        if self.I is None:
            raise ValueError('I has not been set (CoronalIrradtion)')
        result = np.copy(self.I)
        return result

@njit
def time_dep_update_impl(theta, dt, Gamma, GammaPrev, n, nPrev):
    Nlevel = n.shape[0]
    Nspace = n.shape[1]

    GammaPrev = GammaPrev if GammaPrev is not None else np.empty_like(Gamma)
    Gam = np.zeros((Nlevel, Nlevel))
    nk = np.zeros(Nlevel)
    nPrevIter = np.zeros(Nlevel)
    nCurrent = np.zeros(Nlevel)
    atomDelta = 0.0

    for k in range(Nspace):
        nCurrent[:] = n[:, k]
        nPrevIter[:] = nPrev[:, k]
        Gam[...] = -theta * Gamma[:,:, k] * dt
        Gam += np.eye(Nlevel)
        if theta != 1.0:
            nk[:] = (1.0 - theta) * dt * GammaPrev[:,:, k] @ nPrevIter + nPrevIter
        else:
            nk[:] = nPrevIter

        nNew = np.linalg.solve(Gam, nk)
        n[:, k] = nNew
        atomDelta = max(atomDelta, np.nanmax(np.abs(1.0 - nCurrent / nNew)))

    return atomDelta

class MsLightweaverManager:

    def __init__(self, atmost, outputDir,
                 atoms, activeAtoms=['H', 'Ca'],
                 detailedH=False,
                 detailedHPath=None,
                 startingCtx=None, conserveCharge=False,
                 populationTransportMode='Advect',
                 downgoingRadiation=None,
                 prd=False):
        # check_write_git_revision(outputDir)
        self.atmost = atmost
        self.outputDir = outputDir
        self.conserveCharge = conserveCharge
        self.abund = DefaultAtomicAbundance
        self.idx = 0
        self.nHTot = atmost.d1 / (self.abund.massPerH * Const.Amu)
        self.prd = prd
        self.updateRhoPrd = False
        self.detailedH = detailedH
        self.activeAtoms = activeAtoms
        # NOTE(cmo): If this is None and detailedH is True then the data from
        # atmost will be used, otherwise, an MsLw pickle will be loaded from
        # the path.
        self.detailedHPath = detailedHPath
        if populationTransportMode == 'Advect':
            self.advectPops = True
            self.rescalePops = False
        elif populationTransportMode == 'Rescale':
            self.advectPops = False
            self.rescalePops = True
        elif populationTransportMode is None or populationTransportMode == 'None':
            self.advectPops = False
            self.rescalePops = False
        else:
            raise ValueError('Unknown populationTransportMode: %s' % populationTransportMode)

        self.downgoingRadiation = downgoingRadiation

        if startingCtx is not None:
            self.ctx = startingCtx
            args = startingCtx.arguments
            self.atmos = args['atmos']
            self.spect = args['spect']
            self.aSet = self.spect.radSet
            self.eqPops = args['eqPops']
            self.upperBc = self.atmos.upperBc
        else:
            nHTot = np.copy(self.nHTot[0])
            if self.downgoingRadiation:
                self.upperBc = CoronalIrraditation()
            else:
                self.upperBc = None

            self.atmos = Atmosphere.make_1d(scale=ScaleType.Geometric, depthScale=np.copy(atmost.z1[0]), temperature=np.copy(atmost.tg1[0]), vlos=np.copy(atmost.vz1[0]), vturb=np.copy(atmost.vturb), ne=np.copy(atmost.ne1[0]), nHTot=nHTot, upperBc=self.upperBc)

            # self.atmos.convert_scales()
            self.atmos.quadrature(5)

            self.aSet = RadiativeSet(atoms)
            self.aSet.set_active(*activeAtoms)
            if detailedH:
                self.aSet.set_detailed_static('H')
            # NOTE(cmo): Radyn seems to compute the collisional rates once per
            # timestep(?) and we seem to get a much better agreement for Ca
            # with the CH rates when H is set to LTE for the initial timestep.
            # Might be a bug in my implementation though.

            self.spect = self.aSet.compute_wavelength_grid()

            self.mols = MolecularTable()
            if self.conserveCharge:
                self.eqPops = self.aSet.iterate_lte_ne_eq_pops(self.atmos, self.mols)
            else:
                self.eqPops = self.aSet.compute_eq_pops(self.atmos, self.mols)

            nrHOnly = 'He' in activeAtoms
            self.ctx = lw.Context(self.atmos, self.spect, self.eqPops, initSol=InitialSolution.Lte, 
                                  conserveCharge=self.conserveCharge, Nthreads=12, nrHOnly=nrHOnly)

        self.atmos.bHeat = np.ones_like(self.atmost.bheat1[0]) * 1e-20
        self.atmos.hPops = self.eqPops['H']
        np.save(self.outputDir + 'Wavelength.npy', self.ctx.spect.wavelength)
        if self.detailedH:
            self.eqPops['H'][:] = self.detailed_hydrogen_pops()

        if self.downgoingRadiation:
            self.upperBc.set_bc(self.downgoingRadiation.compute_downgoing_radiation(self.spect.wavelength, self.atmos))
        self.ctx.depthData.fill = True
        # self.opac_background()

        # NOTE(cmo): Set up background
        # self.opc = OpcFile('opctab_cmo_mslw.dat')
        # # self.opc = OpcFile()
        # opcWvl = self.opc.wavel
        # self.opcWvl = opcWvl
        # # NOTE(cmo): Find mapping from wavelength array to opctab array, with
        # # constant background over the region of each line. Are overlaps a
        # # problem here? Probably -- but let's see the spectrum in practice
        # # The record to be used is the one in self.wvlIdxs + 4 due to the data
        # # layout in the opctab
        # self.wvlIdxs = np.ones_like(self.spect.wavelength, dtype=np.int64) * -1
        # lineCores = []
        # for a in self.aSet.activeSet:
        #     for l in a.lines:
        #         lineCores.append(l.lambda0 * 10)
        # lineCores = np.array(lineCores)
        # lineCoreIdxs = np.zeros_like(lineCores)
        # for i, l in enumerate(lineCores):
        #     closestIdx = np.argmin(np.abs(opcWvl - l))
        #     lineCoreIdxs[i] = closestIdx

        # for a in self.aSet.activeSet:
        #     for l in a.lines:
        #         # closestIdx = np.argmin((opcWvl - l.lambda0*10)**2)
        #         closestCore = np.argmin(np.abs((l.wavelength * 10)[:, None] - lineCores), axis=1)
        #         closestIdx = lineCoreIdxs[closestCore]
        #         sub = find_subarray(self.spect.wavelength, l.wavelength)
        #         self.wvlIdxs[sub:sub + l.wavelength.shape[0]] = closestIdx
        # for i, v in enumerate(self.wvlIdxs):
        #     if v >= 0:
        #         continue

        #     closestIdx = np.argmin(np.abs(opcWvl - self.spect.wavelength[i]*10))
        #     self.wvlIdxs[i] = closestIdx
        # self.opctabIdxs = self.wvlIdxs + 4

        # NOTE(cmo): Compute initial background opacity
        # np.save('chi.npy', self.ctx.background.chi)
        # np.save('eta.npy', self.ctx.background.eta)
        # np.save('sca.npy', self.ctx.background.sca)
        # self.opac_background()

    def initial_stat_eq(self, Nscatter=3, NmaxIter=1000, popTol=1e-3, JTol=3e-3):
        if self.prd:
            self.ctx.update_hprd_coeffs()

        lw.iterate_ctx_se(self.ctx, prd=self.prd, Nscatter=Nscatter, NmaxIter=NmaxIter, 
                          popsTol=popTol, JTol=JTol)

    def advect_pops(self):
        if self.rescalePops:
            adv = self.atmost.d1[self.idx+1] / self.atmost.d1[self.idx]
            neAdv = self.atmos.ne * adv
            self.atmos.ne[:] = neAdv
            for atom in self.aSet.activeAtoms:
                p = self.eqPops[atom.element]
                for i in range(p.shape[0]):
                    pAdv = p[i] * adv
                    p[i, :] = pAdv
        elif self.advectPops:
            nr_advect(self.atmost, self.idx, self.eqPops, [a.element for a in self.aSet.activeAtoms], self.abund)
            # NOTE(cmo): Guess advected n_e. Will be corrected to be self
            # consistent later (in update_deps if conserveCharge=True). If
            # conserveCharge isn't true then we're using loaded n_e anyway
            # neAdv = interp1d(z0Tracer, np.log10(self.atmos.ne), kind=3, fill_value='extrapolate')(z1)
            # self.atmos.ne[:] = 10**neAdv

    def detailed_hydrogen_pops(self):
        if not self.detailedH:
            raise ValueError('Detailed H pops called without detailedH==True')
        if self.detailedHPath:
            with open(self.detailedHPath + '/Step_%.6d.pickle' % self.idx, 'rb') as pkl:
                step = pickle.load(pkl)
            pops = step['eqPops']['H']['n']
        else:
            pops = self.atmost.nh1[self.idx, :] / (np.sum(self.atmost.nh1[self.idx, :], axis=0) / self.atmos.nHTot)[None, :]
        return pops

    def detailed_ne(self):
        if not self.detailedH:
            raise ValueError('Detailed ne called without detailedH==True')
        if self.detailedHPath:
            with open(self.detailedHPath + '/Step_%.6d.pickle' % self.idx, 'rb') as pkl:
                step = pickle.load(pkl)
            ne = step['ne']
        else:
            ne = self.atmost.ne1[self.idx]
        return ne


    def save_timestep(self):
        i = self.idx
        with open(self.outputDir + 'Step_%.6d.pickle' % i, 'wb') as pkl:
            eqPops = distill_pops(self.eqPops)
            Iwave = self.ctx.spect.I
            lines = []
            for a in self.aSet.activeAtoms:
                lines += self.aSet[a.element].lines
            losses = compute_radiative_losses(self.ctx)
            lineLosses = integrate_line_losses(self.ctx, losses, lines, extendGridNm=5.0)
            pickle.dump({'eqPops': eqPops, 'Iwave': Iwave,
                         'ne': self.atmos.ne, 'lines': lines,
                         'losses': lineLosses}, pkl)

    def load_timestep(self, stepNum):
        with open(self.outputDir + 'Step_%.6d.pickle' % stepNum, 'rb') as pkl:
            step = pickle.load(pkl)

        self.idx = stepNum
        self.atmos.temperature[:] = self.atmost.tg1[self.idx]
        self.atmos.vlos[:] = self.atmost.vz1[self.idx]
        if not self.conserveCharge:
            self.atmos.ne[:] = self.detailed_ne()

        if self.advectPops or self.rescalePops:
            self.atmos.nHTot[:] = self.nHTot[self.idx]
        self.atmos.bHeat[:] = self.atmost.bheat1[self.idx]

        self.atmos.height[:] = self.atmost.z1[self.idx]

        for name, pops in step['eqPops'].items():
            if pops['n'] is not None:
                self.eqPops.atomicPops[name].pops[:] = pops['n']
            self.eqPops.atomicPops[name].nStar[:] = pops['nStar']
        self.atmos.ne[:] = step['ne']
        self.ctx.spect.I[:] = step['Iwave']
        self.ctx.update_deps()

    def increment_step(self):
        self.advect_pops()
        self.idx += 1
        self.atmos.temperature[:] = self.atmost.tg1[self.idx]
        self.atmos.vlos[:] = self.atmost.vz1[self.idx]
        if not self.conserveCharge:
            self.atmos.ne[:] = self.detailed_ne()

        if self.advectPops or self.rescalePops:
            self.atmos.nHTot[:] = self.nHTot[self.idx]
        self.atmos.bHeat[:] = self.atmost.bheat1[self.idx]

        if self.detailedH:
            self.eqPops['H'][:] = self.detailed_hydrogen_pops()

        self.atmos.height[:] = self.atmost.z1[self.idx]
        self.ctx.update_deps()
        if self.prd:
            self.ctx.update_hprd_coeffs()
            self.updateRhoPrd = False
            self.interp_rho_prd()
        if self.downgoingRadiation:
            self.upperBc.set_bc(self.downgoingRadiation.compute_downgoing_radiation(self.spect.wavelength, self.atmos))
        # self.opac_background()

    def interp_rho_prd(self):
        prevIdx = self.idx - 1
        prevZ = self.atmost.z1[prevIdx]
        z = self.atmost.z1[self.idx]

        for atom in self.ctx.activeAtoms:
            for trans in atom.trans:
                try:
                    trans.rhoPrd
                    for la in range(trans.rhoPrd.shape[0]):
                        trans.rhoPrd[la, :] = weno4(z, prevZ, trans.rhoPrd[la])
                    trans.rhoPrd[trans.rhoPrd < 0] = 1e-5
                except AttributeError:
                    pass

    def time_dep_prev_state(self, evalGamma=False):
        if evalGamma:
            self.ctx.formal_sol_gamma_matrices()
        s = {}
        s['pops'] = [np.copy(a.n) for a in self.ctx.activeAtoms]
        s['Gamma'] = [np.copy(a.Gamma) if evalGamma else None for a in self.ctx.activeAtoms]
        return s

    def time_dep_update(self, dt, prevState, theta=1.0):
        atoms = self.ctx.activeAtoms

        maxDelta = 0.0
        for i, atom in enumerate(atoms):
            atomDelta = time_dep_update_impl(theta, dt, atom.Gamma, prevState['Gamma'][i],
                                             atom.n, prevState['pops'][i])

            maxDelta = max(maxDelta, atomDelta)
            s = '    %s delta = %6.4e' % (atom.atomicModel.element, atomDelta)
            print(s)

        return maxDelta

    def time_dep_step(self, nSubSteps=200, popsTol=1e-3, JTol=3e-3, rhoTol=1e-2, dt=None):
        dt = dt if dt is not None else self.atmost.dt[self.idx+1]
        dNrPops = 0.0
        # self.ctx.spect.J[:] = 0.0
        if self.prd:
            for atom in self.ctx.activeAtoms:
                for t in atom.trans:
                    t.recompute_gII()

        prevState = None
        prdStartedOnSub = 0
        for sub in range(nSubSteps):
            if self.updateRhoPrd and sub > 0:
                dPrd = self.ctx.prd_redistribute(maxIter=10, tol=popsTol)
                print(dPrd.compact_representation())

            dJ = self.ctx.formal_sol_gamma_matrices()
            print(dJ.compact_representation())
            delta, prevState = self.ctx.time_dep_update(dt, prevState)
            if self.conserveCharge:
                hOnly = 'He' in self.activeAtoms
                dNrPops = self.ctx.nr_post_update(timeDependentData={'dt': dt, 'nPrev': prevState}, 
                                                  hOnly=hOnly)

            popsChange = dNrPops if self.conserveCharge else delta
            print(popsChange.compact_representation())

            if sub > 1 and (popsChange.dPopsMax < popsTol and dJ.dJMax < JTol):
                if self.prd: 
                    if self.updateRhoPrd:
                        if dPrd.dRhoMax < rhoTol and sub - prdStartedOnSub > 1:
                            break
                    else:
                        print('Starting PRD Iterations')
                        self.updateRhoPrd = True
                        prdStartedOnSub = sub
                else:
                    break
        else:
            raise ValueError('NON-CONVERGED')


    def cont_fn_data(self, step):
        self.load_timestep(step)
        self.ctx.depthData.fill = True
        dJ = 1.0
        while dJ > 1e-5:
            dJ = self.ctx.formal_sol_gamma_matrices()
        self.ctx.depthData.fill = False
        J = np.copy(self.ctx.spect.J)

        sourceData = {'chi': np.copy(self.ctx.depthData.chi),
                      'eta': np.copy(self.ctx.depthData.eta),
                      'chiBg': np.copy(self.ctx.background.chi),
                      'etaBg': np.copy(self.ctx.background.eta),
                      'scaBg': np.copy(self.ctx.background.sca),
                      'J': J
                      }
        return sourceData

    def rf_k(self, step, dt, pertSize, k, Jstart=None):
        self.load_timestep(step)
        print(pertSize)

        self.ctx.clear_ng()
        if Jstart is not None:
            self.ctx.spect.J[:] = Jstart
        else:
            self.ctx.spect.J[:] = 0.0

        if Jstart is None:
            dJ = 1.0
            while dJ > 1e-3:
                dJ = self.ctx.formal_sol_gamma_matrices()
            Jstart = np.copy(self.ctx.spect.J)
                
        self.atmos.temperature[k] += 0.5 * pertSize
        self.ctx.update_deps()

        self.time_dep_step(popsTol=1e-4, JTol=5e-3, dt=dt, theta=1.0)
        plus = np.copy(self.ctx.spect.I[:, -1])

        self.load_timestep(step)
        self.ctx.clear_ng()
        if Jstart is not None:
            self.ctx.spect.J[:] = Jstart
        else:
            self.ctx.spect.J[:] = 0.0
        
        self.atmos.temperature[k] -= 0.5 * pertSize
        self.ctx.update_deps()

        # if Jstart is None:
            # dJ = 1.0
            # while dJ > 1e-3:
                # dJ = self.ctx.formal_sol_gamma_matrices()
        self.time_dep_step(popsTol=1e-4, JTol=5e-3, dt=dt, theta=1.0)
        minus = np.copy(self.ctx.spect.I[:, -1])

        return plus, minus
    
    def rf_k_stat_eq(self, step, dt, pertSize, k, Jstart=None):
        self.load_timestep(step)
        print(pertSize)

        self.ctx.clear_ng()
        if Jstart is not None:
            self.ctx.spect.J[:] = Jstart
        else:
            self.ctx.spect.J[:] = 0.0

        if Jstart is None:
            dJ = 1.0
            while dJ > 1e-3:
                dJ = self.ctx.formal_sol_gamma_matrices()
            Jstart = np.copy(self.ctx.spect.J)
                
        self.atmos.temperature[k] += 0.5 * pertSize
        self.ctx.update_deps()

        # self.time_dep_step(popsTol=1e-4, JTol=5e-3, dt=dt, theta=1.0)
        while True:
            self.ctx.formal_sol_gamma_matrices()
            dPops = self.ctx.stat_equil()
            if dPops < 1e-5 and dPops != 0.0: 
                break
            
        plus = np.copy(self.ctx.spect.I[:, -1])

        self.load_timestep(step)
        self.ctx.clear_ng()
        if Jstart is not None:
            self.ctx.spect.J[:] = Jstart
        else:
            self.ctx.spect.J[:] = 0.0
        
        self.atmos.temperature[k] -= 0.5 * pertSize
        self.ctx.update_deps()

        # if Jstart is None:
            # dJ = 1.0
            # while dJ > 1e-3:
                # dJ = self.ctx.formal_sol_gamma_matrices()
        # self.time_dep_step(popsTol=1e-4, JTol=5e-3, dt=dt, theta=1.0)
        while True:
            self.ctx.formal_sol_gamma_matrices()
            dPops = self.ctx.stat_equil()
            if dPops < 1e-5 and dPops != 0.0:
                break
            
        minus = np.copy(self.ctx.spect.I[:, -1])

        return plus, minus
    
    def rf_ne_k(self, step, dt, pertSizePercent, k, Jstart=None):
        self.load_timestep(step)
        print(pertSizePercent)

        self.ctx.clear_ng()
        if Jstart is not None:
            self.ctx.spect.J[:] = Jstart
        else:
            self.ctx.spect.J[:] = 0.0

        if Jstart is None:
            dJ = 1.0
            while dJ > 1e-3:
                dJ = self.ctx.formal_sol_gamma_matrices()
            Jstart = np.copy(self.ctx.spect.J)
                
        self.atmos.ne[k] += 0.5 * pertSizePercent * self.atmos.ne[k]
        self.ctx.update_deps()

        self.time_dep_step(popsTol=1e-3, JTol=5e-3, dt=dt, theta=1.0)
        plus = np.copy(self.ctx.spect.I[:, -1])

        self.load_timestep(step)
        self.ctx.clear_ng()
        if Jstart is not None:
            self.ctx.spect.J[:] = Jstart
        else:
            self.ctx.spect.J[:] = 0.0
        
        self.atmos.ne[k] -= 0.5 * pertSizePercent * self.atmos.ne[k]
        self.ctx.update_deps()

        # if Jstart is None:
            # dJ = 1.0
            # while dJ > 1e-3:
                # dJ = self.ctx.formal_sol_gamma_matrices()
        self.time_dep_step(popsTol=1e-3, JTol=5e-3, dt=dt, theta=1.0)
        minus = np.copy(self.ctx.spect.I[:, -1])

        return plus, minus
    
    def rf_vlos_k(self, step, dt, pertSize, k, Jstart=None):
        self.load_timestep(step)
        print(pertSize)

        self.ctx.clear_ng()
        if Jstart is not None:
            self.ctx.spect.J[:] = Jstart
        else:
            self.ctx.spect.J[:] = 0.0

        if Jstart is None:
            dJ = 1.0
            while dJ > 1e-3:
                dJ = self.ctx.formal_sol_gamma_matrices()
            Jstart = np.copy(self.ctx.spect.J)
                
        self.atmos.vlos[k] += 0.5 * pertSize
        self.ctx.update_deps()

        self.time_dep_step(popsTol=1e-3, JTol=5e-3, dt=dt, theta=1.0)
        plus = np.copy(self.ctx.spect.I[:, -1])

        self.load_timestep(step)
        self.ctx.clear_ng()
        if Jstart is not None:
            self.ctx.spect.J[:] = Jstart
        else:
            self.ctx.spect.J[:] = 0.0
        
        self.atmos.vlos[k] -= 0.5 * pertSize
        self.ctx.update_deps()

        # if Jstart is None:
            # dJ = 1.0
            # while dJ > 1e-3:
                # dJ = self.ctx.formal_sol_gamma_matrices()
        self.time_dep_step(popsTol=1e-3, JTol=5e-3, dt=dt, theta=1.0)
        minus = np.copy(self.ctx.spect.I[:, -1])

        return plus, minus

def convert_atomic_pops(atom):
    d = {}
    if atom.pops is not None:
        d['n'] = atom.pops
    else:
        d['n'] = atom.pops
    d['nStar'] = atom.nStar
    d['radiativeRates'] = atom.radiativeRates
    return d

def distill_pops(eqPops):
    d = {}
    for atom in eqPops.atomicPops:
        d[atom.element.name] = convert_atomic_pops(atom)
    return d
