from dataclasses import dataclass

import lightweaver as lw
import numpy as np
from lightweaver.atomic_model import (AtomicLevel, AtomicLine, AtomicModel,
                                      ExplicitContinuum, LinearCoreExpWings,
                                      LineType, VoigtLine)
from lightweaver.atomic_set import SpeciesStateTable
from lightweaver.broadening import (LineBroadening, RadiativeBroadening,
                                    VdwApprox)
from lightweaver.collisional_rates import (CI, Ar85Cdi, Burgess,
                                           CollisionalRates, Omega, fone)
from weno4 import weno4


@dataclass(eq=False, repr=False)
class VdwRadyn(VdwApprox):
    def setup(self, line: AtomicLine):
        self.line = line
        if len(self.vals) != 1:
            raise ValueError('VdwRadyn expects 1 coefficient (%s)' % repr(line))

        Z = line.jLevel.stage + 1
        j = line.j
        ic = j + 1
        while line.atom.levels[ic].stage < Z:
            ic += 1
        cont = line.atom.levels[ic]

        zz = line.iLevel.stage + 1
        deltaR = (lw.ERydberg / (cont.E_SI - line.jLevel.E_SI))**2 \
                - (lw.ERydberg / (cont.E_SI - line.iLevel.E_SI))**2
        fourPiEps0 = 4.0 * np.pi * lw.Epsilon0
        c625 = (2.5 * lw.QElectron**2 / fourPiEps0 * lw.ABarH / fourPiEps0 \
                * 2 * np.pi * (Z * lw.RBohr)**2 / lw.HPlanck * deltaR)**0.4

        self.cross = self.vals[0] * 8.411 * (8.0 * lw.KBoltzmann / np.pi * \
            (1.0 / (lw.PeriodicTable['H'].mass * lw.Amu) + \
                1.0 / (line.atom.element.mass * lw.Amu)))**0.3 * c625


    def broaden(self, atmos: lw.Atmosphere, eqPops: SpeciesStateTable) -> np.ndarray:
        nHGround = eqPops['H'][0, :]
        return self.cross * atmos.temperature**0.3 * nHGround

@dataclass
class Shull82(CollisionalRates):
    row: int
    col: int
    #  = 3, 1 for the Ca rates in Radyn
    #  Not used anymore as we pulled the Radyn formulation rather than the RH one
    aCol: float
    tCol: float
    aRad: float
    xRad: float
    aDi: float
    bDi: float
    t0: float
    t1: float

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.iLevel = atom.levels[self.i]
        self.jLevel = atom.levels[self.j]

    def __repr__(self):
        s = 'Shull82(j=%d, i=%d, row=%d, col=%d, aCol=%e, tCol=%e, aRad=%e, xRad=%e, aDi=%e, bDi=%e, t0=%e, t1=%e)' % (self.j, self.i, self.row, self.col, self.aCol, self.tCol, self.aRad, self.xRad, self.aDi, self.bDi, self.t0, self.t1)
        return s

    def compute_rates(self, atmos, eqPops, Cmat):
        nstar = eqPops.atomicPops[self.atom.element].nStar
        # NOTE(cmo): Summers direct recombination rates
        zz = self.jLevel.stage
        iso_seq = self.atom.element.Z - self.iLevel.stage

        rhoq = (atmos.ne * lw.CM_TO_M**3) / zz**7
        # x = (0.5 * zz + (self.col - 1)) * self.row / 3
        # beta = -0.2 / np.log(x + np.e)

        tg = atmos.temperature
        # NOTE(cmo): This is the RH formulation
        # rho0 = 30.0 + 50.0*x
        # y = (1.0 + rhoq/rho0)**beta
        # summersScaling = 1.0
        # summers = summersScaling * y + (1.0 - summersScaling)
        rho0 = 2000.0
        if (iso_seq == lw.PeriodicTable['Li'].Z 
            or iso_seq == lw.PeriodicTable['Na'].Z 
            or iso_seq == lw.PeriodicTable['K'].Z):
            rho0 = 30.0
        summers = 1.0 / (1.0 + rhoq / rho0)**0.14


        cDown = self.aRad * (tg * 1e-4)**(-self.xRad)
        cDown += summers * self.aDi / tg / np.sqrt(tg) * np.exp(-self.t0 / tg) * (1.0 + self.bDi * np.exp(-self.t1 / tg))

        cUp = self.aCol * np.sqrt(tg) * np.exp(-self.tCol / tg) / (1.0 + 0.1 * tg / self.tCol)

        # NOTE(cmo): Rates are in cm-3 s-1, so use ne in cm-3
        cDown *= atmos.ne * lw.CM_TO_M**3
        cUp *= atmos.ne * lw.CM_TO_M**3

        # NOTE(cmo): 3-body recombination (high density limit)
        cDown += cUp * nstar[self.i, :] / nstar[self.j, :]

        Cmat[self.i, self.j, :] += cDown
        Cmat[self.j, self.i, :] += cUp


@dataclass
class Ar85Ch(CollisionalRates):
    t1: float
    t2: float
    a: float
    b: float
    c: float
    d: float

    # AR85-CHH in RH

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.iLevel = atom.levels[self.i]
        self.jLevel = atom.levels[self.j]

    def __repr__(self):
        s = f'Ar85Ch(j={self.j}, i={self.i}, t1={self.t1}, t2={self.t2}, a={self.a}, b={self.b}, c={self.c}, d={self.d})' 
        return s

    def compute_rates(self, atmos, eqPops, Cmat):
        mask = (atmos.temperature >= self.t1) & (atmos.temperature <= self.t2)
        t4 = atmos.temperature * 1e-4
        cDown = self.a * 1e-9 * t4**self.b * (1.0 + self.c*np.exp(self.d * t4)) * eqPops['H'][0] * lw.CM_TO_M**3

        Cmat[self.i, self.j, mask] += cDown[mask]

@dataclass
class Ar85Chp(CollisionalRates):
    t1: float
    t2: float
    a: float
    b: float
    c: float
    d: float

    # AR85-CH+ in Radyn

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.iLevel = atom.levels[self.i]
        self.jLevel = atom.levels[self.j]

    def __repr__(self):
        s = f'Ar85Ch(j={self.j}, i={self.i}, t1={self.t1}, t2={self.t2}, a={self.a}, b={self.b}, c={self.c}, d={self.d})' 
        return s

    def compute_rates(self, atmos, eqPops, Cmat):
        mask = (atmos.temperature >= self.t1) & (atmos.temperature <= self.t2)
        t4 = atmos.temperature * 1e-4
        cUp = self.a * 1e-9 * t4**self.b * np.exp(-self.c * t4) * np.exp(-self.d * lw.EV / lw.KBoltzmann / atmos.temperature) * eqPops['H'][-1] * lw.CM_TO_M**3

        Cmat[self.j, self.i, mask] += cUp[mask]


def hepop(t, toth):
    # Ratio of helium ionisation fractions, from AR85.
    temperature = np.array([3.50, 4.00, 4.10, 4.20, 4.30, 4.40, 4.50, 4.60, 
                   4.70, 4.80, 4.90, 5.00, 5.10, 5.20, 5.30, 5.40, 
                   5.50, 5.60, 5.70])
    one = np.array([0.0, 0.0 ,0.0 ,0.0 ,0.0 ,-0.07,-0.51,-1.33,-2.07, 
                    -2.63,-3.20,-3.94,-4.67,-5.32,-5.90,-6.42,-6.90,-7.33,-7.73])
    two = np.array([-20.0,-9.05,-6.10,-3.75,-2.12,-0.84,-0.16, 
                    -0.02,-0.01,-0.05,-0.34,-0.96,-1.60,-2.16,-2.63,-3.03,-3.38,-3.68,-3.94])
    abhe=0.1
    tlog=np.log10(t)
    f1=10.**(weno4(tlog,temperature,one))
    f2=10.**(weno4(tlog,temperature,two))
    he1=toth*abhe*f1
    he2=toth*abhe*f2
    alfa=1.-f1-f2
    alfa[alfa < 0.0]=1.e-30
    alfa=toth*abhe*alfa
    return he1, he2, alfa

@dataclass
class Ar85Che(CollisionalRates):
    t1: float
    t2: float
    a: float
    b: float
    c: float
    d: float

    # AR85-CHE in Radyn. There's a bit of a mixup over whether these are up or down rates. But the paper is clear that the non-p (i.e. with non-ionised H/He) is recombination.

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.iLevel = atom.levels[self.i]
        self.jLevel = atom.levels[self.j]

    def __repr__(self):
        s = f'Ar85Ch(j={self.j}, i={self.i}, t1={self.t1}, t2={self.t2}, a={self.a}, b={self.b}, c={self.c}, d={self.d})' 
        return s

    def compute_rates(self, atmos, eqPops, Cmat):
        mask = (atmos.temperature >= self.t1) & (atmos.temperature <= self.t2)
        t4 = atmos.temperature * 1e-4
        toth = atmos.nHTot
        he1, _, _ = hepop(atmos.temperature, toth)
        cDown = self.a * 1e-9 * t4**self.b * (1.0 + self.c*np.exp(self.d * t4)) * he1 * lw.CM_TO_M**3

        Cmat[self.i, self.j, mask] += cDown[mask]

@dataclass
class Ar85Chep(CollisionalRates):
    t1: float
    t2: float
    a: float
    b: float
    c: float
    d: float

    # AR85-CHE+/CHP in Radyn. There's a bit of a mixup over whether these are up or down rates. But the paper is clear that the non-p (i.e. with non-ionised H/He) is recombination.

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.iLevel = atom.levels[self.i]
        self.jLevel = atom.levels[self.j]

    def __repr__(self):
        s = f'Ar85Ch(j={self.j}, i={self.i}, t1={self.t1}, t2={self.t2}, a={self.a}, b={self.b}, c={self.c}, d={self.d})' 
        return s

    def compute_rates(self, atmos, eqPops, Cmat):
        mask = (atmos.temperature >= self.t1) & (atmos.temperature <= self.t2)
        t4 = atmos.temperature * 1e-4
        toth = atmos.nHTot
        he1, he2, _ = hepop(atmos.temperature, toth)

        cUp = self.a * 1e-9 * t4**self.b * np.exp(-self.c * t4) * np.exp(-self.d * lw.EV / lw.KBoltzmann / atmos.temperature) * he2 * lw.CM_TO_M**3

        Cmat[self.j, self.i, mask] += cUp[mask]

@dataclass
class Ar85Cea(CollisionalRates):
    fudge: float = 1.0

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.iLevel = atom.levels[self.i]
        self.jLevel = atom.levels[self.j]

    def __repr__(self):
        s = 'Ar85Cea(j=%d, i=%d, fudge=%e)' % (self.j, self.i, self.fudge)
        return s

    def compute_rates(self, atmos, eqPops, Cmat):
        zz = self.atom.element.Z
        isoseq = zz - self.iLevel.stage
        kBT = lw.KBoltzmann * atmos.temperature / lw.EV
        cup = 0.0
        
        if isoseq == lw.PeriodicTable['Li'].Z:
            iea = 13.6 * (zz - 0.835)**2 - 0.25 * (zz - 1.62)**2
            b = 1.0 / (1.0 + 2e-4 * zz**3)
            zeff = zz - 0.43
            y = iea / kBT
            f1y = fone(y)
            g = 2.22*f1y + 0.67*(1.0 - y*f1y) + 0.49*y*f1y + 1.2*y*(1.0 - y*f1y)
            cup = (1.60E-07 * 1.2 * b) / (zeff**2 * np.sqrt(kBT)) * np.exp(-y)*g

            if self.atom.element == lw.PeriodicTable['C']:
                cup *= 0.6
            elif self.atom.element == lw.PeriodicTable['N']:
                cup *= 0.8
            elif self.atom.element == lw.PeriodicTable['O']:
                cup *= 1.25
        elif isoseq == lw.PeriodicTable['Na'].Z:
            if zz <= 16:
                iea = 26.0 * (zz - 10.0)
                a = 2.8e-17 * (zz - 11.0)**(-0.7)
                y = iea / kBT
                f1y = fone(y)
                cup = 6.69e7 * a * iea / np.sqrt(kBT) * np.exp(-y) * (1.0 - y*f1y)
            elif zz >= 18 and zz <= 28:
                iea = 11.0 * (zz - 10.0) * np.sqrt(zz - 10.0)
                a = 1.3e-14 * (zz - 10.0)**(-3.73)
                y = iea / kBT
                f1y = fone(y)
                cup = 6.69e7 * a * iea / np.sqrt(kBT) * np.exp(-y) * (1.0 - 0.5*(y - y*y + y*y*y*f1y))
            else:
                cup = 0.0


        if any(isoseq == lw.PeriodicTable[x].Z for x in ['Mg', 'Al', 'Si', 'P', 'S']):

            if isoseq == lw.PeriodicTable['Mg'].Z:
                iea = 10.3 * (zz - 10.0)**1.52
            if isoseq == lw.PeriodicTable['Al'].Z:
                iea = 18.0 * (zz - 11.0)**1.33
            if isoseq == lw.PeriodicTable['Si'].Z:
                iea = 18.4 * (zz - 12.0)**1.36
            if isoseq == lw.PeriodicTable['P'].Z:
                iea = 23.7 * (zz - 13.0)**1.29
            if isoseq == lw.PeriodicTable['S'].Z:
                iea = 40.1 * (zz - 14.0)**1.1

            a = 4.0e-13 / (zz**2 * iea)
            y = iea / kBT
            f1y = fone(y)
            cup = 6.69e7 * a * iea / np.sqrt(kBT) * np.exp(-y) * (1.0 - 0.5*(y - y*y + y*y*y*f1y) )

        if self.atom.element == lw.PeriodicTable['Ca'] and self.iLevel.stage == 0:
            iea = 25.0
            a = 9.8e-17
            b = 1.12
            cup = 6.69e7 * a * iea / np.sqrt(kBT) * np.exp(-y)*(1.0 + b*f1y)
        elif self.atom.element == lw.PeriodicTable['Ca'] and self.iLevel.stage == 1:
            iea = 25.0
            a = 6.0e-17
            b = 1.12
            cup = 6.69e7 * a * iea / np.sqrt(kBT) * np.exp(-y)*(1.0 + b*f1y)
        elif self.atom.element == lw.PeriodicTable['Fe'] and self.iLevel.stage == 3:
            iea = 60.0
            a = 1.8e-17
            b = 1.0
            cup = 6.69e7 * a * iea / np.sqrt(kBT) * np.exp(-y)*(1.0 + b*f1y)
        elif self.atom.element == lw.PeriodicTable['Fe'] and self.iLevel.stage == 4:
            iea = 73.0
            a = 5.0e-17
            b = 1.12
            cup = 6.69e7 * a * iea / np.sqrt(kBT) * np.exp(-y)*(1.0 + b*f1y)
            

        # NOTE(cmo): From old CaII version
        # a = 6.0e-17 # NOTE(cmo): From looking at the AR85 paper, (page 430), should this instead be 9.8e-17 (Ca+)
        # iea = 25.0
        # NOTE(cmo): From Appendix A to AR85 for Ca+
        # a = 9.8e-17
        # iea = 29.0
        # NOTE(cmo): Changed above back for consistency, need to look into which is technically more correct though
        # y = iea / kBT
        # f1y = fone(y)
        # b = 1.12
        # cUp = 6.69e7 * a * iea / np.sqrt(kBT) * np.exp(-y)*(1.0 + b*f1y)
        # NOTE(cmo): Rates are in cm-3 s-1, so use ne in cm-3
        cup *= self.fudge * atmos.ne * lw.CM_TO_M**3
        Cmat[self.j, self.i, :] += cup

Si_30_gkerr_update = lambda: AtomicModel(element=lw.Element(Z=14),
	levels=[
		AtomicLevel(E=     0.000, g=1, label="si i 3s2 3p2 3pe 0", stage=0, J=None, L=None, S=None),
		AtomicLevel(E=    77.115, g=3, label="si i 3s2 3p2 3pe 1", stage=0, J=None, L=None, S=None),
		AtomicLevel(E=   223.157, g=5, label="si i 3s2 3p2 3pe 2", stage=0, J=None, L=None, S=None),
		AtomicLevel(E=  6298.850, g=5, label="si i 3s2 3p2 1de 2", stage=0, J=None, L=None, S=None),
		AtomicLevel(E= 15394.370, g=1, label="si i 3s2 3p2 1se 0", stage=0, J=None, L=None, S=None),
		AtomicLevel(E= 79664.000, g=3, label="si i 3s 3p3 3so 1", stage=0, J=None, L=None, S=None),
		AtomicLevel(E= 65741.706, g=2, label="si ii 3s2 3p 2po 1/2", stage=1, J=None, L=None, S=None),
		AtomicLevel(E= 66028.906, g=4, label="si ii 3s2 3p 2po 3/2", stage=1, J=None, L=None, S=None),
		AtomicLevel(E=108566.006, g=2, label="si ii 3s 3p2 4pe 1/2", stage=1, J=None, L=None, S=None),
		AtomicLevel(E=108674.306, g=4, label="si ii 3s 3p2 4pe 3/2", stage=1, J=None, L=None, S=None),
		AtomicLevel(E=108849.606, g=6, label="si ii 3s 3p2 4pe 5/2", stage=1, J=None, L=None, S=None),
		AtomicLevel(E=121051.106, g=4, label="si ii 3s 3p2 2de 3/2", stage=1, J=None, L=None, S=None),
		AtomicLevel(E=121066.906, g=6, label="si ii 3s 3p2 2de 5/2", stage=1, J=None, L=None, S=None),
		AtomicLevel(E=142407.106, g=2, label="si ii 3s 3p2 2se 1/2", stage=1, J=None, L=None, S=None),
		AtomicLevel(E=145080.206, g=4, label="si ii 3s2 3d 2de 3/2", stage=1, J=None, L=None, S=None),
		AtomicLevel(E=145096.706, g=6, label="si ii 3s2 3d 2de 5/2", stage=1, J=None, L=None, S=None),
		AtomicLevel(E=149543.706, g=2, label="si ii 3s 3p2 2pe 1/2", stage=1, J=None, L=None, S=None),
		AtomicLevel(E=149746.006, g=4, label="si ii 3s 3p2 2pe 3/2", stage=1, J=None, L=None, S=None),
		AtomicLevel(E=197571.928, g=1, label="si iii 3s2 1se 0", stage=2, J=None, L=None, S=None),
		AtomicLevel(E=250296.628, g=1, label="si iii 3s 3p 3po 0", stage=2, J=None, L=None, S=None),
		AtomicLevel(E=250425.228, g=3, label="si iii 3s 3p 3po 1", stage=2, J=None, L=None, S=None),
		AtomicLevel(E=250686.928, g=5, label="si iii 3s 3p 3po 2", stage=2, J=None, L=None, S=None),
		AtomicLevel(E=280456.328, g=3, label="si iii 3s 3p 1po 1", stage=2, J=None, L=None, S=None),
		AtomicLevel(E=327280.428, g=1, label="si iii 3p2 3pe 0", stage=2, J=None, L=None, S=None),
		AtomicLevel(E=327413.928, g=3, label="si iii 3p2 3pe 1", stage=2, J=None, L=None, S=None),
		AtomicLevel(E=327672.428, g=5, label="si iii 3p2 3pe 2", stage=2, J=None, L=None, S=None),
		AtomicLevel(E=467700.900, g=2, label="si iv 3s 2se 1/2", stage=3, J=None, L=None, S=None),
		AtomicLevel(E=538988.439, g=2, label="si iv 3p 2po 1/2", stage=3, J=None, L=None, S=None),
		AtomicLevel(E=539449.541, g=4, label="si iv 3p 2po 3/2", stage=3, J=None, L=None, S=None),
		AtomicLevel(E=831784.603, g=1, label="si v 2p6 1se 0", stage=4, J=None, L=None, S=None),
	],
	lines=[
		VoigtLine(j=5, i=0, f=4.385e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=8.56667, qWing=42.8333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=5.61e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=5, i=1, f=4.403e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=8.56667, qWing=42.8333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=5.61e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=5, i=2, f=4.456e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=8.56667, qWing=42.8333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=5.61e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=8, i=6, f=4.557e-06, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=4.28333, qWing=17.1333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=10200)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=9, i=6, f=1.032e-08, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=4.28333, qWing=17.1333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1330)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=11, i=6, f=9.324e-04, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=4.28333, qWing=17.1333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.05e+06)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=13, i=6, f=4.420e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=8.56667, qWing=21.4167, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=3.45e+08)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=14, i=6, f=5.825e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=8.56667, qWing=21.4167, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.34e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=16, i=6, f=2.903e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=8.56667, qWing=42.8333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.7e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=17, i=6, f=1.455e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=8.56667, qWing=42.8333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.19e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=8, i=7, f=1.905e-06, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=4.28333, qWing=17.1333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=10200)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=9, i=7, f=1.095e-06, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=4.28333, qWing=17.1333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1330)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=10, i=7, f=2.223e-06, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=4.28333, qWing=17.1333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1810)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=11, i=7, f=4.647e-05, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=4.28333, qWing=17.1333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.05e+06)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=12, i=7, f=4.175e-04, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=4.28333, qWing=17.1333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=562000)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=13, i=7, f=2.201e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=25.7, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=3.45e+08)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=14, i=7, f=2.903e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=25.7, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.34e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=15, i=7, f=2.613e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=25.7, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=7.27e+08)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=16, i=7, f=3.618e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=25.7, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.7e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=17, i=7, f=1.813e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=8.56667, qWing=42.8333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.19e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=20, i=18, f=2.486e-05, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=25.7, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=15400)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=21, i=18, f=3.427e-11, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=8.56667, qWing=42.8333, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=0.0131)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=22, i=18, f=1.748e+00, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=8.56667, qWing=42.8333, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=2.67e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=24, i=19, f=5.777e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=25.7, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=2.28e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=23, i=20, f=1.920e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=25.7, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=2.27e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=24, i=20, f=1.444e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=25.7, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=2.28e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=25, i=20, f=2.403e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=25.7, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=2.28e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=24, i=21, f=1.444e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=25.7, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=2.28e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=25, i=21, f=4.325e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=25.7, Nlambda=30), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=2.28e+09)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=27, i=26, f=2.689e-01, type=LineType.PRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=128.5, Nlambda=90), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=9.12e+08)], elastic=[VdwRadyn(vals=[0.0])])),
		VoigtLine(j=28, i=26, f=5.417e-01, type=LineType.PRD, quadrature=LinearCoreExpWings(qCore=6.425, qWing=128.5, Nlambda=90), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=9.3e+08)], elastic=[VdwRadyn(vals=[0.0])])),
	],
	continua=[
		ExplicitContinuum(j=7, i=0, wavelengthGrid=[39.88, 42.85, 47.33, 50.4, 53.702999999999996, 58.435, 62.55800000000001, 67.61, 72.0, 73.6, 75.0, 78.0, 82.0, 84.0, 87.49, 91.17, 91.2, 94.977, 97.256, 99.0, 101.0, 104.42, 107.0, 110.016, 110.093, 110.1, 115.0, 118.0, 119.9, 120.1, 121.84, 123.881, 123.978, 124.0, 126.5, 129.45499999999998, 132.0, 133.5, 135.0, 136.0, 138.0, 139.375, 140.277, 142.0, 144.0, 146.0, 148.0, 149.0, 150.0, 151.449], alphaGrid=[3.42e-23, 4.08e-23, 5.260000000000001e-23, 6.2e-23, 7.32e-23, 9.15e-23, 1.29e-22, 2.2e-22, 3.78e-22, 3.77e-22, 3.2e-22, 5.1e-22, 6.460000000000001e-22, 7.23e-22, 8.55e-22, 1.08e-21, 1.08e-21, 1.0900000000000001e-21, 1.46e-21, 9.68e-22, 1.4500000000000002e-21, 2.1e-21, 6.56e-22, 1.4200000000000002e-21, 1.43e-21, 1.43e-21, 2.05e-21, 2.35e-21, 2.54e-21, 2.56e-21, 2.74e-21, 2.9400000000000003e-21, 2.95e-21, 2.9600000000000004e-21, 3.2000000000000005e-21, 3.51e-21, 3.75e-21, 3.85e-21, 3.94e-21, 4.0000000000000004e-21, 4.13e-21, 4.22e-21, 4.27e-21, 4.38e-21, 4.5e-21, 4.57e-21, 4.61e-21, 4.6300000000000004e-21, 4.65e-21, 4.67e-21]),
		ExplicitContinuum(j=7, i=1, wavelengthGrid=[36.28, 39.88, 42.85, 47.33, 50.4, 53.702999999999996, 58.435, 62.55800000000001, 67.61, 72.0, 75.0, 78.0, 82.0, 84.0, 87.49, 91.17, 91.2, 94.977, 97.256, 99.0, 101.0, 104.42, 107.0, 110.016, 110.093, 110.1, 115.0, 118.0, 119.9, 120.1, 121.84, 123.881, 123.978, 124.0, 126.5, 129.45499999999998, 130.927, 133.5, 135.0, 136.0, 138.0, 139.375, 140.277, 142.0, 144.0, 146.0, 148.0, 149.0, 150.0, 151.626], alphaGrid=[2.72e-23, 3.42e-23, 4.08e-23, 5.260000000000001e-23, 6.2e-23, 7.32e-23, 9.15e-23, 1.29e-22, 2.2e-22, 3.78e-22, 3.2e-22, 5.1e-22, 6.460000000000001e-22, 7.23e-22, 8.55e-22, 1.08e-21, 1.08e-21, 1.0900000000000001e-21, 1.46e-21, 9.68e-22, 1.4500000000000002e-21, 2.1e-21, 6.56e-22, 1.4200000000000002e-21, 1.43e-21, 1.43e-21, 2.05e-21, 2.35e-21, 2.54e-21, 2.56e-21, 2.74e-21, 2.9400000000000003e-21, 2.95e-21, 2.9600000000000004e-21, 3.2000000000000005e-21, 3.51e-21, 3.65e-21, 3.85e-21, 3.94e-21, 4.0000000000000004e-21, 4.13e-21, 4.22e-21, 4.27e-21, 4.38e-21, 4.5e-21, 4.57e-21, 4.61e-21, 4.6300000000000004e-21, 4.65e-21, 4.67e-21]),
		ExplicitContinuum(j=7, i=2, wavelengthGrid=[36.28, 39.88, 42.85, 47.33, 50.4, 53.702999999999996, 58.435, 62.55800000000001, 67.61, 72.0, 75.0, 78.0, 82.0, 84.0, 87.49, 91.17, 91.2, 94.977, 97.256, 99.0, 101.0, 104.42, 107.0, 110.016, 110.093, 110.1, 115.0, 118.0, 119.9, 120.1, 123.881, 123.978, 124.0, 126.5, 129.45499999999998, 130.927, 133.5, 136.0, 138.0, 140.277, 142.0, 144.0, 146.0, 148.0, 149.0, 150.0, 151.0, 151.793, 151.9, 151.962, 151.9624068103837], alphaGrid=[2.72e-23, 3.42e-23, 4.08e-23, 5.260000000000001e-23, 6.2e-23, 7.32e-23, 9.15e-23, 1.29e-22, 2.2e-22, 3.78e-22, 3.2e-22, 5.1e-22, 6.460000000000001e-22, 7.23e-22, 8.55e-22, 1.08e-21, 1.08e-21, 1.0900000000000001e-21, 1.46e-21, 9.68e-22, 1.4500000000000002e-21, 2.1e-21, 6.56e-22, 1.4200000000000002e-21, 1.43e-21, 1.43e-21, 2.05e-21, 2.35e-21, 2.54e-21, 2.56e-21, 2.9400000000000003e-21, 2.95e-21, 2.9600000000000004e-21, 3.2000000000000005e-21, 3.51e-21, 3.65e-21, 3.85e-21, 4.0000000000000004e-21, 4.13e-21, 4.27e-21, 4.38e-21, 4.5e-21, 4.57e-21, 4.61e-21, 4.6300000000000004e-21, 4.65e-21, 4.66e-21, 4.67e-21, 4.67e-21, 4.67e-21, 4.67e-21]),
		ExplicitContinuum(j=7, i=3, wavelengthGrid=[42.85, 47.33, 50.4, 53.702999999999996, 58.435, 62.55800000000001, 67.61, 70.75, 76.0, 80.0, 82.0, 84.0, 86.03, 88.0, 91.17, 91.2, 92.0, 94.977, 97.256, 101.0, 106.0, 110.016, 110.093, 110.1, 114.0, 116.0, 119.9, 120.1, 123.881, 123.978, 124.0, 129.45499999999998, 132.0, 136.0, 138.0, 142.0, 144.0, 148.0, 150.0, 151.793, 151.9, 152.458, 152.5, 154.906, 158.0, 160.0, 162.0, 164.033, 165.728, 167.42000000000002], alphaGrid=[5.770000000000001e-23, 7.39e-23, 8.679999999999999e-23, 1.02e-22, 1.28e-22, 1.55e-22, 2.3900000000000003e-22, 3.2e-22, 4.62e-22, 7.1400000000000005e-22, 6.060000000000001e-22, 6.5900000000000005e-22, 9.68e-22, 8.230000000000001e-22, 1.2e-21, 1.19e-21, 1.0600000000000001e-21, 1.71e-21, 2.9000000000000004e-21, 9.590000000000001e-22, 1.72e-21, 2.2700000000000004e-21, 2.28e-21, 2.29e-21, 3.0900000000000003e-21, 3.6800000000000004e-21, 5.29e-21, 5.3700000000000006e-21, 6.240000000000001e-21, 6.240000000000001e-21, 6.23e-21, 5.2500000000000005e-21, 4.72e-21, 3.9800000000000006e-21, 3.67e-21, 3.21e-21, 3.0399999999999998e-21, 2.77e-21, 2.67e-21, 2.6100000000000004e-21, 2.6100000000000004e-21, 2.59e-21, 2.5800000000000004e-21, 2.51e-21, 2.42e-21, 2.39e-21, 2.37e-21, 2.35e-21, 2.33e-21, 2.32e-21]),
		ExplicitContinuum(j=7, i=4, wavelengthGrid=[42.85, 47.33, 51.589999999999996, 56.17999999999999, 60.36, 65.0, 70.0, 75.0, 78.0, 82.0, 84.0, 91.17, 91.2, 97.256, 99.0, 102.572, 106.0, 110.016, 110.093, 110.1, 115.0, 119.9, 120.1, 123.881, 123.978, 124.0, 132.0, 135.0, 140.277, 144.0, 148.0, 151.793, 151.9, 152.458, 152.5, 158.0, 162.0, 167.42000000000002, 168.2, 168.22899999999998, 168.25, 171.91199999999998, 176.75, 176.85, 180.0, 183.0, 187.0, 190.768, 193.09, 197.494], alphaGrid=[1.0700000000000002e-22, 1.3e-22, 1.56e-22, 1.88e-22, 2.28e-22, 3.28e-22, 4.82e-22, 6.5e-22, 1.4e-21, 8.62e-22, 9.77e-22, 1.6300000000000001e-21, 1.64e-21, 1.73e-21, 2.82e-21, 6.670000000000001e-21, 2.11e-21, 2.71e-21, 2.7200000000000002e-21, 2.7200000000000002e-21, 3.34e-21, 4.3100000000000004e-21, 4.3600000000000006e-21, 5.57e-21, 5.610000000000001e-21, 5.6200000000000006e-21, 8.26e-21, 6.21e-21, 1.4500000000000002e-21, 3.41e-22, 1.0700000000000002e-22, 1.74e-22, 1.7800000000000003e-22, 1.97e-22, 1.98e-22, 4.23e-22, 5.92e-22, 8.07e-22, 8.31e-22, 8.320000000000001e-22, 8.33e-22, 9.47e-22, 1.1e-21, 1.1e-21, 1.2e-21, 1.3e-21, 1.4200000000000002e-21, 1.5400000000000002e-21, 1.61e-21, 1.73e-21]),
		ExplicitContinuum(j=6, i=0, wavelengthGrid=[36.28, 39.88, 42.85, 47.33, 50.4, 53.702999999999996, 58.435, 62.55800000000001, 65.0, 70.0, 76.2, 78.55, 82.0, 86.03, 88.82000000000001, 91.17, 91.2, 94.977, 101.0, 102.572, 107.0, 108.0, 110.016, 110.093, 110.1, 111.0, 116.0, 119.9, 120.1, 123.881, 123.978, 124.0, 126.5, 129.45499999999998, 132.0, 133.5, 136.0, 138.0, 140.277, 142.0, 144.0, 145.0, 146.0, 148.0, 149.0, 150.0, 151.0, 151.793, 151.9, 152.10999999999999, 152.11044264655985], alphaGrid=[8.81e-24, 1.1700000000000001e-23, 1.45e-23, 1.96e-23, 2.36e-23, 2.86e-23, 3.68e-23, 4.5000000000000003e-23, 5.3e-23, 9.14e-23, 1.9200000000000002e-22, 1.64e-22, 2.5900000000000003e-22, 3.3e-22, 3.8600000000000003e-22, 4.31e-22, 4.31e-22, 5.39e-22, 6.490000000000001e-22, 5.090000000000001e-22, 1.0300000000000001e-21, 1.01e-21, 3.52e-22, 3.41e-22, 3.41e-22, 3.79e-22, 9.180000000000002e-22, 1.1200000000000002e-21, 1.13e-21, 1.32e-21, 1.32e-21, 1.32e-21, 1.4500000000000002e-21, 1.61e-21, 1.74e-21, 1.8100000000000002e-21, 1.92e-21, 1.98e-21, 2.05e-21, 2.11e-21, 2.16e-21, 2.2e-21, 2.23e-21, 2.28e-21, 2.29e-21, 2.3000000000000004e-21, 2.31e-21, 2.31e-21, 2.32e-21, 2.32e-21, 2.32e-21]),
		ExplicitContinuum(j=6, i=1, wavelengthGrid=[42.85, 45.31, 49.010000000000005, 52.239999999999995, 56.17999999999999, 58.435, 62.55800000000001, 67.61, 73.6, 76.2, 80.0, 82.0, 86.03, 88.82000000000001, 91.17, 91.2, 94.977, 97.702, 101.0, 102.572, 107.0, 108.0, 110.016, 110.093, 110.1, 111.0, 113.0, 116.0, 119.9, 120.1, 123.881, 123.978, 124.0, 126.5, 129.45499999999998, 132.0, 133.5, 136.0, 138.0, 140.277, 142.0, 144.0, 145.0, 146.0, 148.0, 150.0, 151.0, 151.793, 151.9, 152.28900000000002, 152.28907768571955], alphaGrid=[1.45e-23, 1.72e-23, 2.18e-23, 2.6300000000000004e-23, 3.2700000000000007e-23, 3.68e-23, 4.5000000000000003e-23, 7.04e-23, 1.35e-22, 1.9200000000000002e-22, 1.8999999999999999e-22, 2.5900000000000003e-22, 3.3e-22, 3.8600000000000003e-22, 4.31e-22, 4.31e-22, 5.39e-22, 5.300000000000001e-22, 6.490000000000001e-22, 5.090000000000001e-22, 1.0300000000000001e-21, 1.01e-21, 3.52e-22, 3.41e-22, 3.41e-22, 3.79e-22, 6.9e-22, 9.180000000000002e-22, 1.1200000000000002e-21, 1.13e-21, 1.32e-21, 1.32e-21, 1.32e-21, 1.4500000000000002e-21, 1.61e-21, 1.74e-21, 1.8100000000000002e-21, 1.92e-21, 1.98e-21, 2.05e-21, 2.11e-21, 2.16e-21, 2.2e-21, 2.23e-21, 2.28e-21, 2.3000000000000004e-21, 2.31e-21, 2.31e-21, 2.31e-21, 2.32e-21, 2.32e-21]),
		ExplicitContinuum(j=6, i=2, wavelengthGrid=[39.88, 42.85, 47.33, 50.4, 53.702999999999996, 58.435, 62.55800000000001, 65.0, 70.0, 76.2, 78.55, 82.0, 86.03, 91.17, 91.2, 97.702, 101.0, 102.572, 107.0, 108.0, 110.016, 110.093, 110.1, 111.0, 113.0, 116.0, 119.9, 120.1, 123.881, 123.978, 124.0, 126.5, 128.0, 130.43699999999998, 132.0, 133.5, 136.0, 138.0, 140.277, 142.0, 144.0, 145.0, 146.0, 148.0, 150.0, 151.793, 151.9, 152.458, 152.5, 152.629], alphaGrid=[1.1700000000000001e-23, 1.45e-23, 1.96e-23, 2.36e-23, 2.86e-23, 3.68e-23, 4.5000000000000003e-23, 5.3e-23, 9.14e-23, 1.9200000000000002e-22, 1.64e-22, 2.5900000000000003e-22, 3.3e-22, 4.31e-22, 4.31e-22, 5.300000000000001e-22, 6.490000000000001e-22, 5.090000000000001e-22, 1.0300000000000001e-21, 1.01e-21, 3.52e-22, 3.41e-22, 3.41e-22, 3.79e-22, 6.9e-22, 9.180000000000002e-22, 1.1200000000000002e-21, 1.13e-21, 1.32e-21, 1.32e-21, 1.32e-21, 1.4500000000000002e-21, 1.5300000000000002e-21, 1.6600000000000001e-21, 1.74e-21, 1.8100000000000002e-21, 1.92e-21, 1.98e-21, 2.05e-21, 2.11e-21, 2.16e-21, 2.2e-21, 2.23e-21, 2.28e-21, 2.3000000000000004e-21, 2.31e-21, 2.31e-21, 2.32e-21, 2.32e-21, 2.32e-21]),
		ExplicitContinuum(j=6, i=3, wavelengthGrid=[42.85, 47.33, 50.4, 53.702999999999996, 58.435, 62.55800000000001, 67.61, 72.0, 78.0, 80.0, 84.0, 87.49, 91.17, 91.2, 94.977, 97.702, 101.0, 104.42, 110.016, 110.093, 110.1, 114.0, 116.85999999999999, 119.9, 120.1, 123.881, 123.978, 124.0, 125.527, 128.0, 132.0, 136.0, 140.277, 144.0, 146.0, 148.0, 151.793, 151.9, 152.458, 152.5, 154.906, 156.081, 158.0, 160.0, 162.0, 164.033, 165.728, 167.42000000000002, 168.2, 168.22899999999998], alphaGrid=[1.96e-23, 2.6400000000000002e-23, 3.1900000000000003e-23, 3.8600000000000004e-23, 4.9700000000000007e-23, 6.1e-23, 7.79e-23, 1.16e-22, 1.61e-22, 2.13e-22, 3.59e-22, 3.07e-22, 4.65e-22, 4.630000000000001e-22, 6.11e-22, 4.92e-22, 1.4800000000000002e-21, 4.82e-22, 8.74e-22, 8.790000000000001e-22, 8.8e-22, 1.16e-21, 1.4500000000000002e-21, 1.89e-21, 1.92e-21, 2.73e-21, 2.7500000000000003e-21, 2.76e-21, 3.01e-21, 3.1e-21, 2.7200000000000002e-21, 2.3000000000000004e-21, 1.9099999999999998e-21, 1.67e-21, 1.5700000000000002e-21, 1.5000000000000001e-21, 1.37e-21, 1.37e-21, 1.35e-21, 1.35e-21, 1.31e-21, 1.28e-21, 1.2500000000000001e-21, 1.2200000000000001e-21, 1.2e-21, 1.2e-21, 1.18e-21, 1.17e-21, 1.17e-21, 1.17e-21]),
		ExplicitContinuum(j=6, i=4, wavelengthGrid=[53.702999999999996, 58.435, 62.55800000000001, 67.61, 73.6, 78.0, 86.03, 88.82000000000001, 91.17, 91.2, 94.977, 103.0, 107.0, 110.016, 110.093, 110.1, 112.0, 113.0, 115.0, 116.0, 119.9, 120.1, 123.881, 123.978, 124.0, 129.45499999999998, 133.5, 138.0, 142.0, 148.0, 151.793, 151.9, 152.458, 152.5, 154.906, 160.0, 167.42000000000002, 168.2, 168.22899999999998, 168.25, 171.91199999999998, 176.75, 176.85, 180.0, 185.0, 187.0, 190.768, 195.0, 197.494, 198.62, 198.62024080082406], alphaGrid=[4.46e-23, 5.74e-23, 7.04e-23, 8.88e-23, 1.22e-22, 1.7800000000000003e-22, 3.06e-22, 6.6e-22, 5.65e-22, 5.6100000000000005e-22, 4.95e-22, 8.68e-22, 8.14e-22, 1.85e-21, 1.9000000000000003e-21, 1.9000000000000003e-21, 3.2300000000000005e-21, 3.2000000000000005e-21, 1.2600000000000002e-21, 1.0600000000000001e-21, 1.37e-21, 1.38e-21, 1.62e-21, 1.6300000000000001e-21, 1.6300000000000001e-21, 2.1800000000000003e-21, 2.86e-21, 3.9e-21, 3.910000000000001e-21, 1.04e-21, 2.4300000000000005e-22, 2.32e-22, 1.8399999999999998e-22, 1.81e-22, 7.23e-23, 8.31e-23, 2.35e-22, 2.52e-22, 2.5300000000000003e-22, 2.5300000000000003e-22, 3.33e-22, 4.23e-22, 4.25e-22, 4.74e-22, 5.55e-22, 5.870000000000001e-22, 6.47e-22, 7.1400000000000005e-22, 7.54e-22, 7.720000000000001e-22, 7.720000000000001e-22]),
		ExplicitContinuum(j=18, i=6, wavelengthGrid=[16.369999999999997, 18.0, 18.71, 19.35, 19.94, 20.47, 20.95, 21.39, 22.15, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 29.43, 30.377999999999997, 31.243000000000002, 31.93, 34.21, 36.28, 38.17, 39.88, 41.44, 42.85, 44.14, 45.31, 46.37, 47.33, 48.21, 49.010000000000005, 49.739999999999995, 50.4, 51.589999999999996, 52.239999999999995, 53.702999999999996, 56.17999999999999, 58.435, 59.141, 60.0, 60.141999999999996, 60.36, 62.55800000000001, 64.16, 65.0, 67.61, 70.0, 70.75, 72.0, 73.6, 75.0, 75.85514040930612], alphaGrid=[7.78e-24, 1.0300000000000001e-23, 1.1600000000000001e-23, 1.2799999999999999e-23, 1.4100000000000002e-23, 1.52e-23, 1.63e-23, 1.73e-23, 1.9200000000000001e-23, 2.09e-23, 2.33e-23, 2.5400000000000003e-23, 2.9800000000000005e-23, 3.3500000000000004e-23, 4.5000000000000003e-23, 4.96e-23, 5.380000000000001e-23, 5.640000000000001e-23, 5.87e-23, 6.39e-23, 5.550000000000001e-23, 5.490000000000001e-23, 5.41e-23, 4.69e-23, 5.0600000000000005e-23, 4.92e-23, 4.85e-23, 4.7700000000000004e-23, 4.7900000000000005e-23, 4.2500000000000004e-23, 3.64e-23, 4.23e-23, 4.7499999999999997e-23, 4.08e-23, 5.53e-23, 3.03e-23, 3.88e-23, 4.44e-23, 5.53e-23, 5.56e-23, 5.52e-23, 4.1100000000000003e-23, 5.770000000000001e-23, 6.58e-23, 9.46e-23, 1.28e-22, 1.39e-22, 1.7900000000000002e-22, 2.87e-22, 8.92e-23, 8.92e-23]),
		ExplicitContinuum(j=18, i=7, wavelengthGrid=[16.369999999999997, 18.0, 18.71, 19.35, 19.94, 20.47, 21.39, 22.15, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 29.43, 30.377999999999997, 31.243000000000002, 31.93, 34.21, 36.28, 38.17, 39.88, 41.44, 42.85, 44.14, 45.31, 46.37, 47.33, 48.21, 49.010000000000005, 49.739999999999995, 50.4, 51.589999999999996, 52.239999999999995, 53.702999999999996, 56.17999999999999, 58.435, 59.141, 60.0, 60.141999999999996, 60.36, 62.55800000000001, 64.16, 65.0, 67.61, 70.0, 70.75, 72.0, 73.6, 75.0, 76.0, 76.02075615991245], alphaGrid=[7.78e-24, 1.0300000000000001e-23, 1.1600000000000001e-23, 1.2799999999999999e-23, 1.4100000000000002e-23, 1.52e-23, 1.73e-23, 1.9200000000000001e-23, 2.09e-23, 2.33e-23, 2.5400000000000003e-23, 2.9800000000000005e-23, 3.3500000000000004e-23, 4.5000000000000003e-23, 4.96e-23, 5.380000000000001e-23, 5.640000000000001e-23, 5.87e-23, 6.39e-23, 5.550000000000001e-23, 5.490000000000001e-23, 5.41e-23, 4.69e-23, 5.0600000000000005e-23, 4.92e-23, 4.85e-23, 4.7700000000000004e-23, 4.7900000000000005e-23, 4.2500000000000004e-23, 3.64e-23, 4.23e-23, 4.7499999999999997e-23, 4.08e-23, 5.53e-23, 3.03e-23, 3.88e-23, 4.44e-23, 5.53e-23, 5.56e-23, 5.52e-23, 4.1100000000000003e-23, 5.770000000000001e-23, 6.58e-23, 9.46e-23, 1.28e-22, 1.39e-22, 1.7900000000000002e-22, 2.87e-22, 8.92e-23, 1.3800000000000001e-22, 1.3800000000000001e-22]),
		ExplicitContinuum(j=18, i=8, wavelengthGrid=[29.43, 31.93, 34.21, 36.28, 38.17, 41.44, 42.85, 47.33, 50.4, 52.239999999999995, 53.702999999999996, 58.435, 60.0, 62.55800000000001, 65.0, 67.61, 70.0, 72.0, 73.6, 76.2, 78.0, 80.0, 80.7, 82.0, 82.65, 84.0, 85.0, 86.03, 87.49, 88.0, 88.82000000000001, 90.0, 91.0, 91.17, 91.2, 94.977, 97.256, 99.0, 101.0, 102.572, 104.42, 106.0, 107.0, 108.0, 109.0, 110.016, 110.093, 110.1, 111.0, 112.0, 112.35207473048811], alphaGrid=[3.17e-23, 4.04e-23, 4.9700000000000007e-23, 5.93e-23, 6.45e-23, 5.960000000000001e-23, 6.46e-23, 4.9700000000000007e-23, 4.96e-23, 6.270000000000001e-23, 5.33e-23, 8.47e-23, 1.21e-22, 1.0100000000000001e-22, 1.33e-22, 1.68e-22, 2.0800000000000002e-22, 2.35e-22, 2.3600000000000003e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22]),
		ExplicitContinuum(j=18, i=9, wavelengthGrid=[29.43, 31.93, 34.21, 36.28, 38.17, 41.44, 42.85, 47.33, 50.4, 52.239999999999995, 53.702999999999996, 58.435, 60.0, 62.55800000000001, 65.0, 67.61, 70.0, 72.0, 73.6, 76.2, 78.0, 80.0, 80.7, 82.0, 82.65, 84.0, 85.0, 86.03, 87.49, 88.0, 88.82000000000001, 90.0, 91.0, 91.17, 91.2, 94.977, 97.256, 99.0, 101.0, 102.572, 104.42, 106.0, 107.0, 108.0, 109.0, 110.016, 110.093, 110.1, 111.0, 112.0, 112.48894824205756], alphaGrid=[3.17e-23, 4.04e-23, 4.9700000000000007e-23, 5.93e-23, 6.45e-23, 5.960000000000001e-23, 6.46e-23, 4.9700000000000007e-23, 4.96e-23, 6.270000000000001e-23, 5.33e-23, 8.47e-23, 1.21e-22, 1.0100000000000001e-22, 1.33e-22, 1.68e-22, 2.0800000000000002e-22, 2.35e-22, 2.3600000000000003e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22]),
		ExplicitContinuum(j=18, i=10, wavelengthGrid=[29.43, 31.93, 34.21, 36.28, 38.17, 41.44, 42.85, 47.33, 50.4, 52.239999999999995, 53.702999999999996, 58.435, 60.0, 62.55800000000001, 65.0, 67.61, 70.0, 72.0, 73.6, 76.2, 78.0, 80.0, 80.7, 82.0, 82.65, 84.0, 85.0, 86.03, 87.49, 88.0, 88.82000000000001, 90.0, 91.0, 91.17, 91.2, 94.977, 97.256, 99.0, 101.0, 102.572, 104.42, 106.0, 107.0, 108.0, 109.0, 110.016, 110.093, 110.1, 111.0, 112.0, 112.71120699478533], alphaGrid=[3.17e-23, 4.04e-23, 4.9700000000000007e-23, 5.93e-23, 6.45e-23, 5.960000000000001e-23, 6.46e-23, 4.9700000000000007e-23, 4.96e-23, 6.270000000000001e-23, 5.33e-23, 8.47e-23, 1.21e-22, 1.0100000000000001e-22, 1.33e-22, 1.68e-22, 2.0800000000000002e-22, 2.35e-22, 2.3600000000000003e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.37e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22, 2.3600000000000003e-22]),
		ExplicitContinuum(j=18, i=11, wavelengthGrid=[36.28, 39.88, 44.14, 48.21, 51.589999999999996, 53.702999999999996, 58.435, 60.36, 65.0, 67.61, 70.0, 72.0, 73.6, 76.2, 78.55, 80.0, 80.7, 82.0, 84.0, 85.0, 87.49, 88.0, 88.82000000000001, 90.0, 91.17, 91.2, 92.0, 95.0, 97.256, 97.702, 99.0, 103.7, 106.0, 107.0, 109.0, 110.016, 110.093, 110.1, 111.0, 113.0, 115.0, 119.9, 120.1, 121.84, 123.881, 123.978, 124.0, 125.527, 126.5, 130.43699999999998, 130.68338445188155], alphaGrid=[4.7600000000000003e-23, 5.36e-23, 5.0600000000000005e-23, 5.85e-23, 6.640000000000001e-23, 5.660000000000001e-23, 9.84e-23, 9.56e-23, 1.46e-22, 2.0200000000000002e-22, 1.7000000000000002e-22, 3.3100000000000003e-22, 2.3600000000000003e-22, 3.04e-22, 6.56e-22, 6.14e-22, 4.47e-22, 3.5700000000000003e-22, 5.23e-22, 4.88e-22, 2.3900000000000003e-22, 4.93e-22, 1.21e-21, 9.76e-22, 3.47e-22, 3.3800000000000005e-22, 1.83e-22, 1.03e-22, 1.6600000000000001e-21, 2.0600000000000002e-21, 2.16e-21, 2.92e-22, 1.08e-21, 5.180000000000001e-22, 3.7300000000000005e-22, 4.38e-22, 4.47e-22, 4.4800000000000005e-22, 5.74e-22, 4.46e-22, 4.46e-22, 4.61e-22, 4.61e-22, 4.66e-22, 4.720000000000001e-22, 4.73e-22, 4.73e-22, 4.79e-22, 4.849999999999999e-22, 4.54e-22, 4.54e-22]),
		ExplicitContinuum(j=18, i=12, wavelengthGrid=[36.28, 39.88, 44.14, 48.21, 51.589999999999996, 53.702999999999996, 58.435, 60.36, 65.0, 67.61, 70.0, 72.0, 73.6, 76.2, 78.55, 80.0, 80.7, 82.0, 84.0, 85.0, 87.49, 88.0, 88.82000000000001, 90.0, 91.17, 91.2, 92.0, 95.0, 97.256, 97.702, 99.0, 103.7, 106.0, 107.0, 109.0, 110.016, 110.093, 110.1, 111.0, 113.0, 115.0, 119.9, 120.1, 121.84, 123.881, 123.978, 124.0, 125.527, 126.5, 130.43699999999998, 130.71037349678818], alphaGrid=[4.7600000000000003e-23, 5.36e-23, 5.0600000000000005e-23, 5.85e-23, 6.640000000000001e-23, 5.660000000000001e-23, 9.84e-23, 9.56e-23, 1.46e-22, 2.0200000000000002e-22, 1.7000000000000002e-22, 3.3100000000000003e-22, 2.3600000000000003e-22, 3.04e-22, 6.56e-22, 6.14e-22, 4.47e-22, 3.5700000000000003e-22, 5.23e-22, 4.88e-22, 2.3900000000000003e-22, 4.93e-22, 1.21e-21, 9.76e-22, 3.47e-22, 3.3800000000000005e-22, 1.83e-22, 1.03e-22, 1.6600000000000001e-21, 2.0600000000000002e-21, 2.16e-21, 2.92e-22, 1.08e-21, 5.180000000000001e-22, 3.7300000000000005e-22, 4.38e-22, 4.47e-22, 4.4800000000000005e-22, 5.74e-22, 4.46e-22, 4.46e-22, 4.61e-22, 4.61e-22, 4.66e-22, 4.720000000000001e-22, 4.73e-22, 4.73e-22, 4.79e-22, 4.849999999999999e-22, 4.54e-22, 4.54e-22]),
		ExplicitContinuum(j=18, i=13, wavelengthGrid=[42.85, 47.33, 50.4, 53.702999999999996, 58.435, 60.36, 65.0, 67.61, 72.0, 75.0, 82.0, 85.0, 88.0, 91.17, 91.2, 97.702, 101.0, 102.572, 106.0, 108.0, 109.0, 110.016, 110.093, 110.1, 111.0, 114.0, 116.0, 118.0, 119.9, 120.1, 121.84, 123.881, 123.978, 124.0, 125.527, 132.0, 133.5, 135.0, 138.0, 140.277, 144.0, 146.0, 148.0, 150.0, 151.0, 151.793, 151.9, 152.458, 152.5, 152.65, 154.906, 158.0, 167.42000000000002, 168.2, 168.22899999999998, 168.25, 176.75, 176.85, 180.801, 181.2749436588411], alphaGrid=[6.52e-23, 5.95e-23, 5.63e-23, 5.85e-23, 5.41e-23, 6.43e-23, 1.4300000000000002e-22, 1.62e-22, 2.7e-22, 2.3000000000000003e-22, 5.55e-22, 8.020000000000001e-22, 7.33e-22, 8.93e-22, 8.960000000000001e-22, 2.0900000000000002e-21, 7.43e-22, 7.830000000000001e-22, 1.58e-21, 4.42e-22, 1.3300000000000001e-21, 3.5e-21, 3.65e-21, 3.66e-21, 5.0000000000000005e-21, 1.59e-21, 1.0300000000000001e-21, 1.56e-22, 9.000000000000001e-23, 1.0800000000000002e-22, 1.23e-22, 7.590000000000001e-25, 1.4100000000000001e-25, 2.7599999999999996e-26, 2.23e-23, 2.68e-23, 3.27e-22, 9.13e-22, 3.52e-22, 1.8700000000000002e-21, 4.280000000000001e-22, 7.350000000000001e-23, 1.0000000000000001e-21, 2.8e-21, 3.3200000000000002e-21, 3.15e-21, 3.07e-21, 2.38e-21, 2.31e-21, 2.0600000000000002e-21, 0.0, 3.33e-24, 1.3900000000000001e-24, 1.4000000000000003e-24, 1.4000000000000003e-24, 1.4000000000000003e-24, 1.6400000000000001e-24, 1.6400000000000001e-24, 1.7e-24, 1.7e-24]),
		ExplicitContinuum(j=18, i=14, wavelengthGrid=[49.010000000000005, 52.239999999999995, 56.17999999999999, 58.435, 62.55800000000001, 67.61, 73.6, 78.0, 86.03, 91.17, 91.2, 97.256, 101.0, 104.42, 108.0, 110.016, 110.093, 110.1, 111.0, 115.0, 118.0, 119.9, 120.1, 120.65, 123.881, 123.978, 124.0, 126.5, 128.0, 130.927, 132.0, 133.5, 135.0, 138.0, 139.375, 144.0, 146.0, 148.0, 151.793, 151.9, 152.458, 152.5, 162.0, 167.42000000000002, 168.2, 168.22899999999998, 168.25, 176.75, 176.85, 189.203, 190.5062287726053], alphaGrid=[5.0900000000000006e-23, 6.120000000000001e-23, 5.21e-23, 5.56e-23, 8.040000000000001e-23, 9.000000000000001e-23, 1.7200000000000002e-22, 2.12e-22, 2.9700000000000004e-22, 4.95e-22, 4.96e-22, 4.52e-22, 9.78e-22, 1.07e-21, 1.9099999999999998e-21, 1.17e-21, 1.16e-21, 1.16e-21, 1.23e-21, 3.3800000000000003e-21, 1.95e-21, 9.500000000000001e-21, 1.0200000000000002e-20, 1.17e-20, 6.44e-21, 6.53e-21, 6.550000000000001e-21, 1.1100000000000001e-20, 1.26e-20, 2.8100000000000003e-21, 1.05e-21, 1.07e-21, 4.32e-21, 1.3500000000000001e-20, 1.3e-20, 5.41e-21, 3.46e-21, 2.37e-21, 1.3300000000000001e-21, 1.31e-21, 1.21e-21, 1.2e-21, 3.52e-22, 1.8399999999999998e-22, 1.6700000000000001e-22, 1.6600000000000002e-22, 1.6600000000000002e-22, 5.1199999999999996e-23, 5.04e-23, 2.3e-24, 2.3e-24]),
		ExplicitContinuum(j=18, i=15, wavelengthGrid=[49.010000000000005, 52.239999999999995, 56.17999999999999, 58.435, 62.55800000000001, 67.61, 73.6, 78.0, 86.03, 91.17, 91.2, 97.256, 101.0, 104.42, 108.0, 110.016, 110.093, 110.1, 111.0, 115.0, 118.0, 119.9, 120.1, 120.65, 123.881, 123.978, 124.0, 126.5, 128.0, 130.927, 132.0, 133.5, 135.0, 138.0, 139.375, 144.0, 146.0, 148.0, 151.793, 151.9, 152.458, 152.5, 162.0, 167.42000000000002, 168.2, 168.22899999999998, 168.25, 176.75, 176.85, 189.203, 190.56613043009133], alphaGrid=[5.0900000000000006e-23, 6.120000000000001e-23, 5.21e-23, 5.56e-23, 8.040000000000001e-23, 9.000000000000001e-23, 1.7200000000000002e-22, 2.12e-22, 2.9700000000000004e-22, 4.95e-22, 4.96e-22, 4.52e-22, 9.78e-22, 1.07e-21, 1.9099999999999998e-21, 1.17e-21, 1.16e-21, 1.16e-21, 1.23e-21, 3.3800000000000003e-21, 1.95e-21, 9.500000000000001e-21, 1.0200000000000002e-20, 1.17e-20, 6.44e-21, 6.53e-21, 6.550000000000001e-21, 1.1100000000000001e-20, 1.26e-20, 2.8100000000000003e-21, 1.05e-21, 1.07e-21, 4.32e-21, 1.3500000000000001e-20, 1.3e-20, 5.41e-21, 3.46e-21, 2.37e-21, 1.3300000000000001e-21, 1.31e-21, 1.21e-21, 1.2e-21, 3.52e-22, 1.8399999999999998e-22, 1.6700000000000001e-22, 1.6600000000000002e-22, 1.6600000000000002e-22, 5.1199999999999996e-23, 5.04e-23, 2.3e-24, 2.3e-24]),
		ExplicitContinuum(j=18, i=16, wavelengthGrid=[58.435, 60.36, 65.0, 67.61, 72.0, 76.2, 80.0, 84.0, 88.82000000000001, 91.17, 91.2, 92.0, 94.977, 97.702, 101.0, 102.572, 104.42, 106.0, 110.016, 110.093, 110.1, 119.329, 119.9, 120.1, 121.84, 123.881, 123.978, 124.0, 125.527, 125.76400000000001, 128.0, 129.45499999999998, 130.927, 132.0, 134.0, 135.0, 138.0, 146.0, 150.0, 151.0, 151.793, 151.9, 152.458, 152.5, 154.906, 156.081, 158.0, 160.0, 162.0, 165.0, 167.42000000000002, 168.2, 168.22899999999998, 168.25, 170.0, 171.91199999999998, 174.0, 176.75, 176.85, 185.0, 197.494, 198.62, 198.65, 206.9, 207.1, 208.2109139913611], alphaGrid=[6.5e-23, 7.76e-23, 1.19e-22, 1.49e-22, 2.6500000000000003e-22, 4.43e-22, 3.4000000000000003e-22, 7.420000000000001e-22, 3.7999999999999998e-22, 1.98e-21, 2.0000000000000002e-21, 2.25e-21, 4.92e-22, 8.420000000000001e-22, 8.600000000000001e-21, 6.340000000000001e-21, 1.8700000000000002e-21, 6.12e-22, 1.58e-22, 1.63e-22, 1.63e-22, 1.1800000000000002e-22, 4.79e-22, 6.32e-22, 2.1200000000000003e-21, 1.5100000000000002e-21, 1.4400000000000001e-21, 1.4200000000000002e-21, 2.5000000000000002e-22, 2.9500000000000004e-23, 0.0, 9.66e-21, 1.94e-20, 1.59e-20, 8.270000000000001e-22, 0.0, 1.1e-22, 1.5e-22, 5.56e-22, 9.99e-22, 2.0600000000000002e-21, 2.23e-21, 3.3e-21, 3.39e-21, 7.12e-21, 7.54e-21, 5.100000000000001e-21, 8.369999999999999e-22, 3.5700000000000003e-22, 2.64e-22, 5.180000000000001e-22, 5.440000000000001e-22, 5.42e-22, 5.4e-22, 2.17e-22, 3.1700000000000002e-24, 3.8400000000000003e-23, 3.6000000000000004e-23, 3.59e-23, 2.8300000000000004e-23, 2.4500000000000002e-23, 2.44e-23, 2.44e-23, 2.52e-23, 2.52e-23, 2.52e-23]),
		ExplicitContinuum(j=18, i=17, wavelengthGrid=[53.702999999999996, 58.435, 60.36, 65.0, 70.0, 76.2, 80.0, 84.0, 88.82000000000001, 91.17, 91.2, 95.0, 97.702, 101.0, 102.572, 104.42, 106.0, 110.016, 110.093, 110.1, 119.329, 119.9, 120.1, 121.84, 123.881, 123.978, 124.0, 125.527, 125.76400000000001, 128.0, 129.45499999999998, 130.927, 132.0, 134.0, 135.0, 138.0, 146.0, 150.0, 151.0, 151.793, 151.9, 152.458, 152.5, 154.906, 156.081, 158.0, 160.0, 162.0, 165.0, 167.42000000000002, 168.2, 168.22899999999998, 168.25, 170.0, 171.91199999999998, 174.0, 176.75, 176.85, 185.0, 197.494, 198.62, 198.65, 206.9, 207.1, 209.09163026695015], alphaGrid=[5.78e-23, 6.5e-23, 7.76e-23, 1.19e-22, 1.98e-22, 4.43e-22, 3.4000000000000003e-22, 7.420000000000001e-22, 3.7999999999999998e-22, 1.98e-21, 2.0000000000000002e-21, 4.84e-22, 8.420000000000001e-22, 8.600000000000001e-21, 6.340000000000001e-21, 1.8700000000000002e-21, 6.12e-22, 1.58e-22, 1.63e-22, 1.63e-22, 1.1800000000000002e-22, 4.79e-22, 6.32e-22, 2.1200000000000003e-21, 1.5100000000000002e-21, 1.4400000000000001e-21, 1.4200000000000002e-21, 2.5000000000000002e-22, 2.9500000000000004e-23, 0.0, 9.66e-21, 1.94e-20, 1.59e-20, 8.270000000000001e-22, 0.0, 1.1e-22, 1.5e-22, 5.56e-22, 9.99e-22, 2.0600000000000002e-21, 2.23e-21, 3.3e-21, 3.39e-21, 7.12e-21, 7.54e-21, 5.100000000000001e-21, 8.369999999999999e-22, 3.5700000000000003e-22, 2.64e-22, 5.180000000000001e-22, 5.440000000000001e-22, 5.42e-22, 5.4e-22, 2.17e-22, 3.1700000000000002e-24, 3.8400000000000003e-23, 3.6000000000000004e-23, 3.59e-23, 2.8300000000000004e-23, 2.4500000000000002e-23, 2.44e-23, 2.44e-23, 2.52e-23, 2.52e-23, 2.52e-23]),
		ExplicitContinuum(j=26, i=18, wavelengthGrid=[7.44, 9.1, 10.620000000000001, 12.0, 13.25, 14.39, 15.430000000000001, 16.369999999999997, 17.22, 18.0, 18.71, 19.35, 19.94, 20.47, 20.95, 21.39, 21.79, 22.15, 22.48, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 29.43, 30.377999999999997, 31.243000000000002, 31.93, 34.21, 36.28, 37.019353851463215], alphaGrid=[4.7200000000000006e-24, 8.62e-24, 1.37e-23, 1.91e-23, 2.1200000000000002e-23, 2.3200000000000002e-23, 2.52e-23, 2.65e-23, 2.78e-23, 2.91e-23, 3.07e-23, 3.2100000000000004e-23, 3.33e-23, 3.39e-23, 3.42e-23, 3.4700000000000004e-23, 3.51e-23, 3.54e-23, 3.57e-23, 3.6000000000000004e-23, 3.6300000000000005e-23, 3.7e-23, 3.8900000000000005e-23, 3.9700000000000003e-23, 4.26e-23, 4.32e-23, 4.3000000000000006e-23, 4.33e-23, 4.37e-23, 4.740000000000001e-23, 4.740000000000001e-23]),
		ExplicitContinuum(j=26, i=19, wavelengthGrid=[7.44, 9.1, 10.620000000000001, 12.0, 13.25, 14.39, 15.430000000000001, 16.369999999999997, 17.22, 18.0, 18.71, 19.35, 19.94, 20.47, 20.95, 21.39, 21.79, 22.15, 22.48, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 29.43, 30.377999999999997, 31.243000000000002, 31.93, 34.21, 36.28, 38.17, 39.88, 41.44, 42.85, 44.14, 45.31, 45.997256208470446], alphaGrid=[4.830000000000001e-24, 8.750000000000001e-24, 1.39e-23, 2e-23, 2.52e-23, 2.79e-23, 3.04e-23, 3.2700000000000007e-23, 3.4700000000000004e-23, 3.64e-23, 3.79e-23, 3.92e-23, 4.04e-23, 4.1500000000000005e-23, 4.2400000000000003e-23, 4.33e-23, 4.4100000000000005e-23, 4.47e-23, 4.52e-23, 4.55e-23, 4.6800000000000006e-23, 4.7700000000000004e-23, 4.93e-23, 5.0600000000000005e-23, 5.22e-23, 5.380000000000001e-23, 5.48e-23, 5.36e-23, 5.53e-23, 4.7499999999999997e-23, 4.8299999999999995e-23, 4.98e-23, 2.89e-23, 9.990000000000001e-23, 3.51e-23, 3.3200000000000003e-23, 3.3200000000000003e-23]),
		ExplicitContinuum(j=26, i=20, wavelengthGrid=[7.44, 9.1, 10.620000000000001, 12.0, 13.25, 14.39, 15.430000000000001, 16.369999999999997, 17.22, 18.0, 18.71, 19.35, 19.94, 20.47, 20.95, 21.39, 21.79, 22.15, 22.48, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 29.43, 30.377999999999997, 31.243000000000002, 31.93, 34.21, 36.28, 38.17, 39.88, 41.44, 42.85, 44.14, 45.31, 46.02448082636697], alphaGrid=[4.830000000000001e-24, 8.750000000000001e-24, 1.39e-23, 2e-23, 2.52e-23, 2.79e-23, 3.04e-23, 3.2700000000000007e-23, 3.4700000000000004e-23, 3.64e-23, 3.79e-23, 3.92e-23, 4.04e-23, 4.1500000000000005e-23, 4.2400000000000003e-23, 4.33e-23, 4.4100000000000005e-23, 4.47e-23, 4.52e-23, 4.55e-23, 4.6800000000000006e-23, 4.7700000000000004e-23, 4.93e-23, 5.0600000000000005e-23, 5.22e-23, 5.380000000000001e-23, 5.48e-23, 5.36e-23, 5.53e-23, 4.7499999999999997e-23, 4.8299999999999995e-23, 4.98e-23, 2.89e-23, 9.990000000000001e-23, 3.51e-23, 3.3200000000000003e-23, 3.3200000000000003e-23]),
		ExplicitContinuum(j=26, i=21, wavelengthGrid=[7.44, 9.1, 10.620000000000001, 12.0, 13.25, 14.39, 15.430000000000001, 16.369999999999997, 17.22, 18.0, 18.71, 19.35, 19.94, 20.47, 20.95, 21.39, 21.79, 22.15, 22.48, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 29.43, 30.377999999999997, 31.243000000000002, 31.93, 34.21, 36.28, 38.17, 39.88, 41.44, 42.85, 44.14, 45.31, 46.079982352472676], alphaGrid=[4.830000000000001e-24, 8.750000000000001e-24, 1.39e-23, 2e-23, 2.52e-23, 2.79e-23, 3.04e-23, 3.2700000000000007e-23, 3.4700000000000004e-23, 3.64e-23, 3.79e-23, 3.92e-23, 4.04e-23, 4.1500000000000005e-23, 4.2400000000000003e-23, 4.33e-23, 4.4100000000000005e-23, 4.47e-23, 4.52e-23, 4.55e-23, 4.6800000000000006e-23, 4.7700000000000004e-23, 4.93e-23, 5.0600000000000005e-23, 5.22e-23, 5.380000000000001e-23, 5.48e-23, 5.36e-23, 5.53e-23, 4.7499999999999997e-23, 4.8299999999999995e-23, 4.98e-23, 2.89e-23, 9.990000000000001e-23, 3.51e-23, 3.3200000000000003e-23, 3.3200000000000003e-23]),
		ExplicitContinuum(j=26, i=22, wavelengthGrid=[7.44, 9.1, 10.620000000000001, 12.0, 13.25, 14.39, 15.430000000000001, 16.369999999999997, 17.22, 18.0, 18.71, 19.35, 19.94, 20.47, 20.95, 21.39, 21.79, 22.15, 22.48, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 29.43, 30.377999999999997, 31.243000000000002, 31.93, 34.21, 36.28, 38.17, 39.88, 41.44, 42.85, 44.14, 45.31, 46.37, 47.33, 48.21, 49.010000000000005, 49.739999999999995, 50.4, 51.589999999999996, 52.239999999999995, 53.406087520657195], alphaGrid=[4.160000000000001e-24, 7.55e-24, 1.1900000000000001e-23, 1.72e-23, 2.26e-23, 2.62e-23, 2.86e-23, 3.07e-23, 3.26e-23, 3.41e-23, 3.53e-23, 3.6300000000000005e-23, 3.7300000000000004e-23, 3.82e-23, 3.9e-23, 3.9700000000000003e-23, 4.04e-23, 4.1e-23, 4.1500000000000005e-23, 4.2e-23, 4.3399999999999996e-23, 4.4200000000000006e-23, 4.52e-23, 4.66e-23, 4.95e-23, 4.7100000000000007e-23, 4.63e-23, 5.1600000000000004e-23, 5.000000000000001e-23, 5.53e-23, 4.6800000000000006e-23, 3.7200000000000003e-23, 1.8000000000000002e-23, 3.8400000000000003e-23, 3.7200000000000003e-23, 4.3800000000000004e-23, 1.77e-23, 3.52e-23, 5.15e-23, 4.58e-23, 1.97e-23, 1.1900000000000001e-23, 1.59e-23, 1.59e-23, 1.59e-23]),
		ExplicitContinuum(j=26, i=23, wavelengthGrid=[14.39, 15.430000000000001, 16.369999999999997, 17.22, 18.0, 18.71, 19.35, 19.94, 20.47, 20.95, 21.39, 21.79, 22.15, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 29.43, 30.377999999999997, 31.243000000000002, 31.93, 34.21, 36.28, 38.17, 39.88, 41.44, 42.85, 44.14, 45.31, 46.37, 47.33, 48.21, 49.010000000000005, 49.739999999999995, 50.4, 51.589999999999996, 52.239999999999995, 53.702999999999996, 56.17999999999999, 58.435, 59.141, 60.0, 60.141999999999996, 62.55800000000001, 64.16, 65.0, 67.61, 70.0, 70.75, 71.21468727152545], alphaGrid=[3.12e-23, 3.55e-23, 3.8400000000000003e-23, 4.09e-23, 4.31e-23, 4.49e-23, 4.6400000000000004e-23, 4.78e-23, 4.89e-23, 5.000000000000001e-23, 5.0900000000000006e-23, 5.18e-23, 5.260000000000001e-23, 5.400000000000001e-23, 5.59e-23, 5.74e-23, 6.02e-23, 6.23e-23, 6.330000000000001e-23, 6.56e-23, 6.740000000000001e-23, 6.63e-23, 6.46e-23, 6.570000000000001e-23, 6.380000000000001e-23, 6.34e-23, 6.070000000000001e-23, 6.11e-23, 6.04e-23, 6.19e-23, 5.99e-23, 5.72e-23, 5.63e-23, 6.47e-23, 9.230000000000001e-23, 8.01e-23, 6.61e-23, 7.1e-23, 1.14e-22, 0.0, 2.23e-22, 3.94e-23, 2.42e-25, 0.0, 6.230000000000001e-26, 7.06e-25, 1.63e-24, 5.28e-22, 6.42e-24, 1.62e-24, 1.62e-24]),
		ExplicitContinuum(j=26, i=24, wavelengthGrid=[14.39, 15.430000000000001, 16.369999999999997, 17.22, 18.0, 18.71, 19.35, 19.94, 20.47, 20.95, 21.39, 21.79, 22.15, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 29.43, 30.377999999999997, 31.243000000000002, 31.93, 34.21, 36.28, 38.17, 39.88, 41.44, 42.85, 44.14, 45.31, 46.37, 47.33, 48.21, 49.010000000000005, 49.739999999999995, 50.4, 51.589999999999996, 52.239999999999995, 53.702999999999996, 56.17999999999999, 58.435, 59.141, 60.0, 60.141999999999996, 62.55800000000001, 64.16, 65.0, 67.61, 70.0, 70.75, 71.28245664893242], alphaGrid=[3.12e-23, 3.55e-23, 3.8400000000000003e-23, 4.09e-23, 4.31e-23, 4.49e-23, 4.6400000000000004e-23, 4.78e-23, 4.89e-23, 5.000000000000001e-23, 5.0900000000000006e-23, 5.18e-23, 5.260000000000001e-23, 5.400000000000001e-23, 5.59e-23, 5.74e-23, 6.02e-23, 6.23e-23, 6.330000000000001e-23, 6.56e-23, 6.740000000000001e-23, 6.63e-23, 6.46e-23, 6.570000000000001e-23, 6.380000000000001e-23, 6.34e-23, 6.070000000000001e-23, 6.11e-23, 6.04e-23, 6.19e-23, 5.99e-23, 5.72e-23, 5.63e-23, 6.47e-23, 9.230000000000001e-23, 8.01e-23, 6.61e-23, 7.1e-23, 1.14e-22, 0.0, 2.23e-22, 3.94e-23, 2.42e-25, 0.0, 6.230000000000001e-26, 7.06e-25, 1.63e-24, 5.28e-22, 6.42e-24, 8.99e-25, 8.99e-25]),
		ExplicitContinuum(j=26, i=25, wavelengthGrid=[14.39, 15.430000000000001, 16.369999999999997, 17.22, 18.0, 18.71, 19.35, 19.94, 20.47, 20.95, 21.39, 21.79, 22.15, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 29.43, 30.377999999999997, 31.243000000000002, 31.93, 34.21, 36.28, 38.17, 39.88, 41.44, 42.85, 44.14, 45.31, 46.37, 47.33, 48.21, 49.010000000000005, 49.739999999999995, 50.4, 51.589999999999996, 52.239999999999995, 53.702999999999996, 56.17999999999999, 58.435, 59.141, 60.0, 60.141999999999996, 62.55800000000001, 64.16, 65.0, 67.61, 70.0, 70.75, 71.41404785163976], alphaGrid=[3.12e-23, 3.55e-23, 3.8400000000000003e-23, 4.09e-23, 4.31e-23, 4.49e-23, 4.6400000000000004e-23, 4.78e-23, 4.89e-23, 5.000000000000001e-23, 5.0900000000000006e-23, 5.18e-23, 5.260000000000001e-23, 5.400000000000001e-23, 5.59e-23, 5.74e-23, 6.02e-23, 6.23e-23, 6.330000000000001e-23, 6.56e-23, 6.740000000000001e-23, 6.63e-23, 6.46e-23, 6.570000000000001e-23, 6.380000000000001e-23, 6.34e-23, 6.070000000000001e-23, 6.11e-23, 6.04e-23, 6.19e-23, 5.99e-23, 5.72e-23, 5.63e-23, 6.47e-23, 9.230000000000001e-23, 8.01e-23, 6.61e-23, 7.1e-23, 1.14e-22, 0.0, 2.23e-22, 3.94e-23, 2.42e-25, 0.0, 6.230000000000001e-26, 7.06e-25, 1.63e-24, 5.28e-22, 6.44e-24, 0.0, 0.0]),
		ExplicitContinuum(j=29, i=26, wavelengthGrid=[7.44, 9.1, 10.620000000000001, 12.0, 13.25, 14.39, 15.430000000000001, 16.369999999999997, 17.22, 18.0, 18.71, 19.35, 19.94, 20.47, 20.95, 21.39, 21.79, 22.15, 22.48, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 27.466211526638975], alphaGrid=[7.49e-24, 9.38e-24, 1.1200000000000002e-23, 1.2799999999999999e-23, 1.43e-23, 1.56e-23, 1.69e-23, 1.81e-23, 1.9200000000000001e-23, 2.0100000000000003e-23, 2.1e-23, 2.18e-23, 2.2500000000000002e-23, 2.3200000000000002e-23, 2.3700000000000004e-23, 2.42e-23, 2.4700000000000003e-23, 2.5100000000000002e-23, 2.5400000000000003e-23, 2.5800000000000002e-23, 2.66e-23, 2.73e-23, 2.87e-23, 2.97e-23, 2.97e-23]),
		ExplicitContinuum(j=29, i=27, wavelengthGrid=[7.44, 9.1, 10.620000000000001, 12.0, 13.25, 14.39, 15.430000000000001, 16.369999999999997, 17.22, 18.0, 18.71, 19.35, 19.94, 20.47, 20.95, 21.39, 21.79, 22.15, 22.48, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 29.43, 30.377999999999997, 31.243000000000002, 31.93, 34.1534529120402], alphaGrid=[1.01e-23, 1.2799999999999999e-23, 1.55e-23, 1.7900000000000001e-23, 2.02e-23, 2.2400000000000004e-23, 2.4299999999999998e-23, 2.61e-23, 2.77e-23, 2.92e-23, 3.0600000000000003e-23, 3.1900000000000003e-23, 3.31e-23, 3.42e-23, 3.53e-23, 3.62e-23, 3.7e-23, 3.77e-23, 3.83e-23, 3.88e-23, 4.04e-23, 4.1500000000000005e-23, 4.35e-23, 4.49e-23, 4.8200000000000006e-23, 4.92e-23, 5.000000000000001e-23, 5.0500000000000005e-23, 5.0500000000000005e-23]),
		ExplicitContinuum(j=29, i=28, wavelengthGrid=[7.44, 9.1, 10.620000000000001, 12.0, 13.25, 14.39, 15.430000000000001, 16.369999999999997, 17.22, 18.0, 18.71, 19.35, 19.94, 20.47, 20.95, 21.39, 21.79, 22.15, 22.48, 22.78, 23.64, 24.303, 25.631999999999998, 26.669999999999998, 29.43, 30.377999999999997, 31.243000000000002, 31.93, 34.20732337607864], alphaGrid=[1.01e-23, 1.2799999999999999e-23, 1.55e-23, 1.7900000000000001e-23, 2.02e-23, 2.2400000000000004e-23, 2.4299999999999998e-23, 2.61e-23, 2.77e-23, 2.92e-23, 3.0600000000000003e-23, 3.1900000000000003e-23, 3.31e-23, 3.42e-23, 3.53e-23, 3.62e-23, 3.7e-23, 3.77e-23, 3.83e-23, 3.88e-23, 4.04e-23, 4.1500000000000005e-23, 4.35e-23, 4.49e-23, 4.8200000000000006e-23, 4.92e-23, 5.000000000000001e-23, 5.0500000000000005e-23, 5.0500000000000005e-23]),
	],
	collisions=[
		Omega(j=1, i=0, temperature=[1000.0, 10000000.0], rates=[1.39, 1.39]),
		Omega(j=2, i=0, temperature=[1000.0, 10000000.0], rates=[12.6, 12.6]),
		Omega(j=3, i=0, temperature=[1000.0, 10000000.0], rates=[0.398, 0.398]),
		Omega(j=4, i=0, temperature=[1000.0, 10000000.0], rates=[0.0309, 0.0309]),
		Omega(j=2, i=1, temperature=[1000.0, 10000000.0], rates=[25.3, 25.3]),
		Omega(j=3, i=1, temperature=[1000.0, 10000000.0], rates=[3.74, 3.74]),
		Omega(j=4, i=1, temperature=[1000.0, 10000000.0], rates=[0.199, 0.199]),
		Omega(j=3, i=2, temperature=[1000.0, 10000000.0], rates=[18.2, 18.2]),
		Omega(j=4, i=2, temperature=[1000.0, 10000000.0], rates=[2.19, 2.19]),
		Omega(j=4, i=3, temperature=[1000.0, 10000000.0], rates=[41.1, 41.1]),
		Omega(j=5, i=0, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.95, 2.05, 2.31, 2.94, 4.28, 6.27, 8.79, 11.8, 15.3, 19.1]),
		Omega(j=5, i=1, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[5.85, 6.15, 6.92, 8.82, 12.8, 18.8, 26.4, 35.4, 46.0, 57.3]),
		Omega(j=5, i=2, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[9.78, 10.3, 11.6, 14.7, 21.5, 31.4, 44.0, 59.3, 77.1, 96.0]),
		Omega(j=7, i=6, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[5.57, 5.58, 5.61, 5.74, 5.77, 5.57, 5.32, 5.13, 5.01, 4.94]),
		Omega(j=8, i=6, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.577, 0.563, 0.541, 0.509, 0.467, 0.422, 0.385, 0.36, 0.344, 0.335]),
		Omega(j=9, i=6, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.875, 0.852, 0.817, 0.768, 0.706, 0.639, 0.582, 0.544, 0.521, 0.507]),
		Omega(j=10, i=6, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.604, 0.586, 0.56, 0.528, 0.488, 0.442, 0.402, 0.375, 0.358, 0.347]),
		Omega(j=11, i=6, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[2.8, 2.78, 2.76, 2.72, 2.59, 2.31, 1.96, 1.66, 1.45, 1.32]),
		Omega(j=12, i=6, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[2.45, 2.46, 2.45, 2.43, 2.32, 2.06, 1.78, 1.63, 1.56, 1.52]),
		Omega(j=13, i=6, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.55, 0.642, 0.76, 0.861, 0.902, 0.932, 0.975, 1.02, 1.08, 1.19]),
		Omega(j=14, i=6, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[2.95, 3.06, 3.22, 3.45, 3.75, 4.21, 4.95, 6.07, 7.59, 9.53]),
		Omega(j=15, i=6, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.31, 1.24, 1.16, 1.07, 0.988, 0.914, 0.853, 0.81, 0.783, 0.766]),
		Omega(j=16, i=6, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.78, 1.82, 1.87, 1.95, 2.08, 2.29, 2.51, 2.65, 2.72, 2.77]),
		Omega(j=17, i=6, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.31, 1.32, 1.33, 1.35, 1.39, 1.46, 1.58, 1.78, 2.08, 2.49]),
		Omega(j=8, i=7, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.463, 0.446, 0.423, 0.397, 0.365, 0.33, 0.301, 0.281, 0.268, 0.26]),
		Omega(j=9, i=7, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.2, 1.16, 1.11, 1.04, 0.955, 0.862, 0.784, 0.73, 0.695, 0.674]),
		Omega(j=10, i=7, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[2.41, 2.37, 2.29, 2.16, 1.99, 1.8, 1.64, 1.53, 1.46, 1.41]),
		Omega(j=11, i=7, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[3.53, 3.51, 3.49, 3.47, 3.34, 3.01, 2.59, 2.2, 1.92, 1.75]),
		Omega(j=12, i=7, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[6.99, 6.93, 6.85, 6.74, 6.35, 5.53, 4.61, 3.85, 3.34, 3.03]),
		Omega(j=13, i=7, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.09, 1.28, 1.53, 1.73, 1.83, 1.91, 1.99, 1.98, 1.95, 1.98]),
		Omega(j=14, i=7, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[2.17, 2.14, 2.12, 2.13, 2.21, 2.33, 2.42, 2.39, 2.35, 2.39]),
		Omega(j=15, i=7, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[6.22, 6.35, 6.56, 6.86, 7.27, 7.89, 8.62, 9.37, 10.3, 11.7]),
		Omega(j=16, i=7, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.28, 1.29, 1.32, 1.35, 1.4, 1.51, 1.62, 1.68, 1.71, 1.73]),
		Omega(j=17, i=7, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[4.71, 4.75, 4.81, 4.93, 5.15, 5.53, 6.02, 6.49, 7.05, 7.91]),
		Omega(j=9, i=8, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[5.33, 5.1, 4.8, 4.43, 3.94, 3.42, 3.04, 2.8, 2.65, 2.56]),
		Omega(j=10, i=8, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.68, 1.67, 1.68, 1.67, 1.57, 1.37, 1.19, 1.08, 1.0, 0.958]),
		Omega(j=11, i=8, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.19, 1.2, 1.21, 1.19, 1.09, 0.953, 0.846, 0.778, 0.737, 0.712]),
		Omega(j=12, i=8, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.629, 0.641, 0.653, 0.65, 0.615, 0.552, 0.493, 0.459, 0.442, 0.433]),
		Omega(j=10, i=9, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[7.68, 7.5, 7.26, 6.85, 6.3, 5.79, 5.41, 5.15, 4.99, 4.9]),
		Omega(j=11, i=9, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.81, 1.84, 1.86, 1.84, 1.7, 1.48, 1.31, 1.2, 1.14, 1.1]),
		Omega(j=12, i=9, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.81, 1.84, 1.87, 1.85, 1.72, 1.53, 1.38, 1.28, 1.22, 1.19]),
		Omega(j=11, i=10, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.36, 1.38, 1.4, 1.39, 1.31, 1.18, 1.07, 0.996, 0.951, 0.924]),
		Omega(j=12, i=10, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[4.07, 4.14, 4.19, 4.12, 3.82, 3.34, 2.97, 2.73, 2.59, 2.51]),
		Omega(j=12, i=11, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[6.21, 6.11, 6.0, 5.9, 5.75, 5.52, 5.31, 5.17, 5.08, 5.03]),
		Omega(j=19, i=18, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.883, 0.794, 0.668, 0.527, 0.423, 0.325, 0.198, 0.106, 0.0586, 0.0332]),
		Omega(j=20, i=18, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[2.57, 2.33, 1.98, 1.59, 1.28, 0.969, 0.592, 0.321, 0.178, 0.101]),
		Omega(j=21, i=18, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[4.41, 3.97, 3.34, 2.64, 2.12, 1.63, 0.991, 0.529, 0.291, 0.165]),
		Omega(j=22, i=18, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[4.99, 5.12, 5.33, 5.68, 6.22, 6.95, 7.86, 8.95, 10.5, 12.7]),
		Omega(j=23, i=18, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.0299, 0.0293, 0.0282, 0.0264, 0.0233, 0.0185, 0.0132, 0.00905, 0.00645, 0.00491]),
		Omega(j=24, i=18, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.0888, 0.0873, 0.0847, 0.0797, 0.0698, 0.0551, 0.0396, 0.0273, 0.019, 0.0139]),
		Omega(j=25, i=18, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.149, 0.146, 0.141, 0.133, 0.116, 0.092, 0.0659, 0.0454, 0.0324, 0.0246]),
		Omega(j=20, i=19, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.74, 1.76, 1.79, 1.82, 1.83, 1.8, 1.67, 1.42, 1.14, 0.915]),
		Omega(j=21, i=19, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[3.72, 3.7, 3.67, 3.6, 3.44, 3.17, 2.81, 2.37, 1.93, 1.6]),
		Omega(j=22, i=19, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.962, 0.954, 0.939, 0.916, 0.879, 0.806, 0.678, 0.518, 0.371, 0.262]),
		Omega(j=23, i=19, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.124, 0.123, 0.123, 0.119, 0.111, 0.0981, 0.0862, 0.0743, 0.0637, 0.0559]),
		Omega(j=24, i=19, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[2.1, 2.16, 2.24, 2.35, 2.51, 2.81, 3.28, 3.85, 4.58, 5.52]),
		Omega(j=25, i=19, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.407, 0.403, 0.397, 0.385, 0.362, 0.326, 0.287, 0.25, 0.216, 0.19]),
		Omega(j=21, i=20, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[10.5, 10.5, 10.4, 10.3, 10.1, 9.42, 8.37, 7.08, 5.83, 4.85]),
		Omega(j=22, i=20, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[2.89, 2.86, 2.82, 2.75, 2.63, 2.42, 2.04, 1.55, 1.1, 0.787]),
		Omega(j=23, i=20, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[2.1, 2.14, 2.22, 2.34, 2.52, 2.81, 3.27, 3.91, 4.68, 5.64]),
		Omega(j=24, i=20, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[2.26, 2.3, 2.35, 2.41, 2.48, 2.65, 2.94, 3.31, 3.8, 4.47]),
		Omega(j=25, i=20, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[3.55, 3.61, 3.7, 3.8, 3.95, 4.24, 4.7, 5.32, 6.14, 7.26]),
		Omega(j=22, i=21, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[4.81, 4.77, 4.7, 4.58, 4.38, 4.03, 3.41, 2.58, 1.82, 1.29]),
		Omega(j=23, i=21, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.407, 0.404, 0.398, 0.386, 0.361, 0.326, 0.288, 0.25, 0.217, 0.193]),
		Omega(j=24, i=21, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[3.55, 3.6, 3.68, 3.8, 3.97, 4.25, 4.75, 5.47, 6.38, 7.54]),
		Omega(j=25, i=21, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[9.23, 9.44, 9.74, 10.1, 10.6, 11.6, 13.2, 15.3, 17.9, 21.4]),
		Omega(j=23, i=22, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[0.564, 0.569, 0.569, 0.548, 0.492, 0.407, 0.315, 0.239, 0.185, 0.15]),
		Omega(j=24, i=22, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.69, 1.7, 1.7, 1.65, 1.48, 1.22, 0.946, 0.718, 0.553, 0.446]),
		Omega(j=25, i=22, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[2.82, 2.85, 2.85, 2.74, 2.46, 2.04, 1.58, 1.19, 0.916, 0.739]),
		Omega(j=24, i=23, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.74, 1.73, 1.71, 1.68, 1.62, 1.53, 1.4, 1.22, 1.05, 0.92]),
		Omega(j=25, i=23, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[1.23, 1.25, 1.28, 1.3, 1.32, 1.31, 1.28, 1.24, 1.2, 1.18]),
		Omega(j=25, i=24, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[4.93, 4.97, 5.01, 5.04, 5.01, 4.87, 4.63, 4.31, 4.01, 3.8]),
		Omega(j=27, i=26, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[6.44, 6.46, 6.48, 6.52, 6.6, 6.74, 6.98, 7.36, 7.93, 8.69]),
		Omega(j=28, i=26, temperature=[1000.0, 2511.89, 6309.57, 15848.9, 39810.7, 100000.0, 251189.0, 630958.0, 1584890.0, 3981080.0], rates=[12.8, 12.8, 12.9, 13.0, 13.1, 13.4, 13.9, 14.7, 15.8, 17.3]),
		Shull82(j=6, i=0, row=0, col=0, aCol=0.000000e+00, tCol=9.460000e+04, aRad=0.000000e+00, xRad=6.010000e-01, aDi=1.220000e-04, bDi=0.000000e+00, t0=7.700000e+04, t1=0.000000e+00),
		Shull82(j=7, i=0, row=0, col=0, aCol=0.000000e+00, tCol=9.460000e+04, aRad=0.000000e+00, xRad=6.010000e-01, aDi=1.220000e-04, bDi=0.000000e+00, t0=7.700000e+04, t1=0.000000e+00),
		Shull82(j=6, i=1, row=0, col=0, aCol=0.000000e+00, tCol=9.460000e+04, aRad=0.000000e+00, xRad=6.010000e-01, aDi=3.670000e-04, bDi=0.000000e+00, t0=7.700000e+04, t1=0.000000e+00),
		Shull82(j=7, i=1, row=0, col=0, aCol=0.000000e+00, tCol=9.460000e+04, aRad=0.000000e+00, xRad=6.010000e-01, aDi=3.670000e-04, bDi=0.000000e+00, t0=7.700000e+04, t1=0.000000e+00),
		Shull82(j=6, i=2, row=0, col=0, aCol=0.000000e+00, tCol=9.460000e+04, aRad=0.000000e+00, xRad=6.010000e-01, aDi=6.110000e-04, bDi=0.000000e+00, t0=7.700000e+04, t1=0.000000e+00),
		Shull82(j=7, i=2, row=0, col=0, aCol=0.000000e+00, tCol=9.460000e+04, aRad=0.000000e+00, xRad=6.010000e-01, aDi=6.110000e-04, bDi=0.000000e+00, t0=7.700000e+04, t1=0.000000e+00),
		Shull82(j=18, i=6, row=0, col=0, aCol=0.000000e+00, tCol=1.900000e+05, aRad=0.000000e+00, xRad=7.860000e-01, aDi=1.960000e-03, bDi=7.530000e-01, t0=9.630000e+04, t1=6.460000e+04),
		Shull82(j=18, i=7, row=0, col=0, aCol=0.000000e+00, tCol=1.900000e+05, aRad=0.000000e+00, xRad=7.860000e-01, aDi=3.910000e-03, bDi=7.530000e-01, t0=9.630000e+04, t1=6.460000e+04),
		Shull82(j=26, i=18, row=0, col=0, aCol=0.000000e+00, tCol=3.880000e+05, aRad=0.000000e+00, xRad=6.930000e-01, aDi=5.030000e-03, bDi=1.880000e-01, t0=8.750000e+04, t1=4.710000e+04),
		Shull82(j=29, i=26, row=0, col=0, aCol=0.000000e+00, tCol=5.240000e+05, aRad=0.000000e+00, xRad=8.210000e-01, aDi=5.430000e-03, bDi=4.500000e-01, t0=1.050000e+06, t1=7.980000e+05),
		Ar85Cdi(j=6, i=0, cdi=[[8.1, 24.83, -16.47, 0.43, -18.2], [13.5, 17.93, -11.93, 0.47, -13.57]]),
		Ar85Cdi(j=7, i=0, cdi=[[8.1, 49.67, -32.93, 0.87, -36.4], [13.5, 35.87, -23.87, 0.93, -27.13]]),
		Ar85Cdi(j=6, i=1, cdi=[[8.09, 24.83, -16.47, 0.43, -18.2], [13.49, 17.93, -11.93, 0.47, -13.57]]),
		Ar85Cdi(j=7, i=1, cdi=[[8.09, 49.67, -32.93, 0.87, -36.4], [13.49, 35.87, -23.87, 0.93, -27.13]]),
		Ar85Cdi(j=6, i=2, cdi=[[8.07, 24.83, -16.47, 0.43, -18.2], [13.47, 17.93, -11.93, 0.47, -13.57]]),
		Ar85Cdi(j=7, i=2, cdi=[[8.07, 49.67, -32.93, 0.87, -36.4], [13.47, 35.87, -23.87, 0.93, -27.13]]),
		Ar85Cdi(j=18, i=6, cdi=[[16.3, 50.4, -33.4, 0.6, -36.9], [22.9, 55.1, -37.2, 1.4, -41.0]]),
		Ar85Cdi(j=18, i=7, cdi=[[16.26, 50.4, -33.4, 0.6, -36.9], [22.86, 55.1, -37.2, 1.4, -41.0]]),
		Ar85Cdi(j=26, i=18, cdi=[[33.5, 19.8, -5.7, 1.3, -11.9], [133.0, 66.7, -24.8, 18.7, -65.0], [176.6, 22.0, -7.2, 3.3, -20.9]]),
		Ar85Cea(j=6, i=0, fudge=3.333000e-01),
		Ar85Cea(j=7, i=0, fudge=6.667000e-01),
		Ar85Cea(j=6, i=1, fudge=3.333000e-01),
		Ar85Cea(j=7, i=1, fudge=6.667000e-01),
		Ar85Cea(j=6, i=2, fudge=3.333000e-01),
		Ar85Cea(j=7, i=2, fudge=6.667000e-01),
		Ar85Cea(j=18, i=6, fudge=1.000000e+00),
		Ar85Cea(j=18, i=7, fudge=1.000000e+00),
		Ar85Cea(j=26, i=18, fudge=1.000000e+00),
		Ar85Cea(j=29, i=26, fudge=1.000000e+00),
		Ar85Ch(j=18, i=6, t1=300.0, t2=100000.0, a=5.0, b=0.28, c=0.0, d=0.0),
		Ar85Ch(j=18, i=7, t1=300.0, t2=100000.0, a=5.0, b=0.28, c=0.0, d=0.0),
		Ar85Ch(j=26, i=18, t1=300.0, t2=30000.0, a=0.41, b=0.0, c=0.0, d=0.0),
		Ar85Ch(j=29, i=26, t1=1000.0, t2=30000.0, a=2.4, b=0.0, c=0.0, d=0.0),
		Ar85Ch(j=6, i=0, t1=5000.0, t2=30000.0, a=0.00333, b=0.0, c=0.0, d=0.03),
		Ar85Ch(j=7, i=0, t1=5000.0, t2=30000.0, a=0.00667, b=0.0, c=0.0, d=0.03),
		Ar85Ch(j=6, i=1, t1=5000.0, t2=30000.0, a=0.00333, b=0.0, c=0.0, d=0.03),
		Ar85Ch(j=7, i=1, t1=5000.0, t2=30000.0, a=0.00667, b=0.0, c=0.0, d=0.03),
		Ar85Ch(j=6, i=2, t1=5000.0, t2=30000.0, a=0.00333, b=0.0, c=0.0, d=0.03),
		Ar85Ch(j=7, i=2, t1=5000.0, t2=30000.0, a=0.00667, b=0.0, c=0.0, d=0.03),
		Ar85Ch(j=18, i=6, t1=5000.0, t2=100000.0, a=1.7, b=0.32, c=0.0, d=2.74),
		Ar85Ch(j=18, i=7, t1=5000.0, t2=100000.0, a=1.7, b=0.32, c=0.0, d=2.74),
		Ar85Ch(j=26, i=18, t1=1000.0, t2=30000.0, a=0.95, b=0.75, c=0.0, d=0.0),
		Ar85Ch(j=18, i=6, t1=10000.0, t2=300000.0, a=0.15, b=0.24, c=0.0, d=6.91),
		Ar85Ch(j=18, i=7, t1=10000.0, t2=300000.0, a=0.15, b=0.24, c=0.0, d=6.91),
		Ar85Ch(j=26, i=18, t1=10000.0, t2=300000.0, a=1.15, b=0.44, c=0.0, d=8.88),
		CI(j=6, i=3, temperature=[100.0, 1000000000.0], rates=[1.3500000000000002e-16, 1.3500000000000002e-16]),
		CI(j=7, i=3, temperature=[100.0, 1000000000.0], rates=[2.6700000000000005e-16, 2.6700000000000005e-16]),
		CI(j=6, i=4, temperature=[100.0, 1000000000.0], rates=[1.8800000000000002e-16, 1.8800000000000002e-16]),
		CI(j=7, i=4, temperature=[100.0, 1000000000.0], rates=[3.720000000000001e-16, 3.720000000000001e-16]),
		CI(j=6, i=5, temperature=[100.0, 1000000000.0], rates=[3.6900000000000004e-15, 3.6900000000000004e-15]),
		CI(j=7, i=5, temperature=[100.0, 1000000000.0], rates=[7.700000000000001e-15, 7.700000000000001e-15]),
		Burgess(j=18, i=8, fudge=1),
		Burgess(j=18, i=9, fudge=1),
		Burgess(j=18, i=10, fudge=1),
		Burgess(j=18, i=11, fudge=1),
		Burgess(j=18, i=12, fudge=1),
		Burgess(j=18, i=13, fudge=1),
		Burgess(j=18, i=14, fudge=1),
		Burgess(j=18, i=15, fudge=1),
		Burgess(j=18, i=16, fudge=1),
		Burgess(j=18, i=17, fudge=1),
		Burgess(j=26, i=19, fudge=1),
		Burgess(j=26, i=20, fudge=1),
		Burgess(j=26, i=21, fudge=1),
		Burgess(j=26, i=22, fudge=1),
		Burgess(j=26, i=23, fudge=1),
		Burgess(j=26, i=24, fudge=1),
		Burgess(j=26, i=25, fudge=1),
		Burgess(j=29, i=27, fudge=1),
		Burgess(j=29, i=28, fudge=1),
])
