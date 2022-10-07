import numpy as np
from dataclasses import dataclass
from radynpy.cdf import LazyRadynData
from cdflib import CDF
from typing import Optional
from weno4 import weno4
from tqdm import tqdm

@dataclass
class Atmost:
    grav: float
    tau2: float
    vturb: np.ndarray

    time: np.ndarray
    dt: np.ndarray
    z1: np.ndarray
    d1: np.ndarray
    ne1: np.ndarray
    tg1: np.ndarray
    vz1: np.ndarray
    nh1: np.ndarray
    bheat1: Optional[np.ndarray]

    cgs: bool = True

    def to_SI(self):
        if not self.cgs:
            return

        self.vturb /= 1e2
        self.z1 /= 1e2
        self.d1 *= 1e3
        self.ne1 *= 1e6
        self.vz1 /= 1e2
        self.nh1 *= 1e6

        # NOTE(cmo): we don't change the units on bheat1, since it's only used
        # for the Fang rates, which are entirely described with cgs.

        self.cgs = False

    def reinterpolate(self, maxTimestep=0.01, tol=1e-4) -> 'Atmost':
        '''
        Reinterpolates atmost to not surpass a maxTimestep.
        Tol is a small biasing parameter to prevent just falling short of pre-existing timesteps (which are all kept).

        The implementation is very inefficient for now.
        '''

        time = []
        dt = []
        z1 = []
        d1 = []
        ne1 = []
        tg1 = []
        vz1 = []
        nh1 = []
        bheat = []

        def interp_param(z, t0, t1, alpha, param):
            z0 = self.z1[t0]
            z1 = self.z1[t1]
            p0 = weno4(z, z0, param[t0])
            p1 = weno4(z, z1, param[t1])
            return (1.0 - alpha) * p0 + alpha * p1


        for t in tqdm(range(self.time.shape[0])):
            if ((t != self.time.shape[0] - 1)
                and (self.time[t+1] - self.time[t] > maxTimestep + tol)):
                startTime = self.time[t]
                elapsed = 0.0
                currentTime = startTime + elapsed
                while currentTime < self.time[t+1]:
                    if currentTime == self.time[t]:
                        z = self.z1[t]
                    else:
                        z = 0.5 * (self.z1[t] + self.z1[t+1])

                    alpha = (currentTime - self.time[t]) / (self.time[t+1] - self.time[t])
                    z1.append(z)
                    d1.append(interp_param(z, t, t+1, alpha, self.d1))
                    ne1.append(interp_param(z, t, t+1, alpha, self.ne1))
                    tg1.append(interp_param(z, t, t+1, alpha, self.tg1))
                    vz1.append(interp_param(z, t, t+1, alpha, self.vz1))
                    nh = np.zeros_like(self.nh1[0])
                    for i in range(nh.shape[1]):
                        nh[:, i] = interp_param(z, t, t+1, alpha, self.nh1[:, :, i])
                    nh1.append(nh)
                    if self.bheat1 is not None:
                        bheat.append(interp_param(z, t, t+1, alpha, self.bheat1))
                    time.append(currentTime)

                    elapsed += maxTimestep
                    currentTime += maxTimestep
                    if currentTime + tol >= self.time[t+1]:
                        break

            else:
                time.append(self.time[t])
                z1.append(self.z1[t])
                d1.append(self.d1[t])
                ne1.append(self.ne1[t])
                tg1.append(self.tg1[t])
                vz1.append(self.vz1[t])
                nh1.append(self.nh1[t])
                if self.bheat1 is not None:
                    bheat.append(self.bheat1[t])

        time = np.array(time)
        z1 = np.stack(z1)
        d1 = np.stack(d1)
        ne1 = np.stack(ne1)
        tg1 = np.stack(tg1)
        vz1 = np.stack(vz1)
        nh1 = np.stack(nh1)
        if self.bheat1 is not None:
            bheat = np.stack(bheat)
        else:
            bheat = None

        dt = time[1:] - time[:-1]
        dt = np.concatenate([[dt[0]], dt])

        return Atmost(grav=self.grav, tau2=self.tau2, vturb=self.vturb, 
                      time=time, dt=dt, z1=z1, d1=d1, ne1=ne1, tg1=tg1, 
                      vz1=vz1, nh1=nh1, bheat1=bheat)



def read_cdf(filename) -> Atmost:
    cdf = LazyRadynData(filename)
    grav = cdf.grav
    vturb = np.copy(cdf.vturb)

    time = cdf.time
    dt = cdf.time[1:] - cdf.time[:-1]
    dt = np.concatenate([[dt[0]], dt])
    z1 = np.copy(cdf.z1)
    d1 = np.copy(cdf.d1)
    ne1 = np.copy(cdf.ne1)
    tg1 = np.copy(cdf.tg1)
    vz1 = np.copy(cdf.vz1)
    nh1 = np.copy(cdf.n1[:, :, :6, 0])
    bheat1 = np.copy(cdf.bheat1)

    return Atmost(grav=grav, tau2=0.0, vturb=vturb, time=time, dt=dt,
                  z1=z1, d1=d1, ne1=ne1, tg1=tg1, vz1=vz1, nh1=nh1,
                  bheat1=bheat1)

def read_atmost_cdf(filename) -> Atmost:
    cdf = CDF(filename)
    grav = cdf.varget('grav')
    tau2 = np.copy(cdf.varget('tau2'))
    vturb = np.copy(cdf.varget('vturb'))

    time = np.copy(cdf.varget('time'))
    dt = np.copy(cdf.varget('dtnm'))
    z1 = np.copy(cdf.varget('z1'))
    d1 = np.copy(cdf.varget('d1'))
    ne1 = np.copy(cdf.varget('ne1'))
    tg1 = np.copy(cdf.varget('tg1'))
    vz1 = np.copy(cdf.varget('vz1'))
    nh1 = np.copy(cdf.varget('nh1'))

    return Atmost(grav=grav, tau2=tau2, vturb=vturb, time=time, dt=dt,
                  z1=z1, d1=d1, ne1=ne1, tg1=tg1, vz1=vz1, nh1=nh1,
                  bheat1=None)


def read_atmost(filename='atmost.dat') -> Atmost:
    with open(filename, 'rb') as f:
        # Record: itype 4, isize 4, cname 8 : 16
        _ = np.fromfile(f, np.int32, 1)
        itype = np.fromfile(f, np.int32, 1)
        isize = np.fromfile(f, np.int32, 1)
        cname = np.fromfile(f, 'c', 8)
        _ = np.fromfile(f, np.int32, 1)

        # Record: ntime 4, ndep 4 : 8
        _ = np.fromfile(f, np.int32, 1)
        ntime = np.fromfile(f, np.int32, 1)
        ndep = np.fromfile(f, np.int32, 1)
        _ = np.fromfile(f, np.int32, 1)

        # Record: itype 4, isize 4, cname 8 : 16
        _ = np.fromfile(f, np.int32, 1)
        itype = np.fromfile(f, np.int32, 1)
        isize = np.fromfile(f, np.int32, 1)
        cname = np.fromfile(f, 'c', 8)
        _ = np.fromfile(f, np.int32, 1)

        # Record: grav 8, tau(2) 8, vturb 8 x ndep(300) : 2416
        _ = np.fromfile(f, np.int32, 1)
        grav = np.fromfile(f, np.float64, 1)
        tau2 = np.fromfile(f, np.float64, 1)
        vturb = np.fromfile(f, np.float64, ndep[0])
        _ = np.fromfile(f, np.int32, 1)
        if grav[0] == 0.0:
            grav[0] = 10**4.44

        times = []
        dtns = []
        z1t = []
        d1t = []
        ne1t = []
        tg1t = []
        vz1t = []
        nh1t = []
        bheat1t = []
        while True:
            # Record: itype 4, isize 4, cname 8 : 16
            _ = np.fromfile(f, np.int32, 1)
            itype = np.fromfile(f, np.int32, 1)
            isize = np.fromfile(f, np.int32, 1)
            cname = np.fromfile(f, 'c', 8)
            _ = np.fromfile(f, np.int32, 1)

            # Record: timep 8, dtnp 8, z1 8 * ndep(300),
            # d1 8 * ndep(300), ne1 8 * ndep(300),
            # tg1 8 * ndep(300), vz1 8 * ndep(300),
            # nh1 8 * 6 * ndep(300): 26416
            # bheat1 8 * ndep(300): 26416 + 2400
            recordSize = np.fromfile(f, np.int32, 1)
            if (recordSize - 16) / (8 * ndep[0]) == 11:
                bheat = False
            else:
                bheat = True
            times.append(np.fromfile(f, np.float64, 1))
            if times[-1].shape != (1,):
                times.pop()
                break
            dtns.append(np.fromfile(f, np.float64, 1))
            z1t.append(np.fromfile(f, np.float64, ndep[0]))
            d1t.append(np.fromfile(f, np.float64, ndep[0]))
            ne1t.append(np.fromfile(f, np.float64, ndep[0]))
            tg1t.append(np.fromfile(f, np.float64, ndep[0]))
            vz1t.append(np.fromfile(f, np.float64, ndep[0]))
            nh1t.append(np.fromfile(f, np.float64, ndep[0] * 6).reshape(6, ndep[0]))
            if bheat:
                bheat1t.append(np.fromfile(f, np.float64, ndep[0]))
            _ = np.fromfile(f, np.int32, 1)

    times = np.array(times).squeeze()
    dtns = np.array(dtns).squeeze()
    z1t = np.array(z1t).squeeze()
    d1t = np.array(d1t).squeeze()
    ne1t = np.array(ne1t).squeeze()
    tg1t = np.array(tg1t).squeeze()
    vz1t = np.array(vz1t).squeeze()
    nh1t = np.array(nh1t).squeeze()
    bheat1t = np.array(bheat1t).squeeze()

    return Atmost(grav.item(), tau2.item(), vturb, times, dtns, z1t, d1t, ne1t, tg1t, vz1t, nh1t, bheat1t)

def read_flarix(filename, filenameHPops, Ntime, Ndepth) -> Atmost:
    z1t = np.zeros((Ntime, Ndepth))
    tg1t = np.zeros((Ntime, Ndepth))
    ne1t = np.zeros((Ntime, Ndepth))
    d1t = np.zeros((Ntime, Ndepth))
    n1t = np.zeros((Ntime, Ndepth))
    vz1t = np.zeros((Ntime, Ndepth))
    nh1t = np.zeros((Ntime, Ndepth, 6))
    with open(filename, 'rb') as f:

        for t in range(Ntime):
            for k in range(Ndepth):
                _ = np.fromfile(f, np.int32, 1)
                z1t[t, k] = np.fromfile(f, np.float64, 1)
                tg1t[t, k] = np.fromfile(f, np.float64, 1)
                ne1t[t, k] = np.fromfile(f, np.float64, 1)
                n1t[t,k] = np.fromfile(f, np.float64, 1)
                d1t[t, k] = np.fromfile(f, np.float64, 1)
                _ = np.fromfile(f, np.float64, 1)
                vz1t[t, k] = np.fromfile(f, np.float64, 1)
                _ = np.fromfile(f, np.float64, 1)
                _ = np.fromfile(f, np.int32, 1)

    with open(filenameHPops, 'rb') as f:
        for t in range(Ntime):
            _ = np.fromfile(f, np.int32, 1)
            nh1t[t].reshape(-1)[...] = np.fromfile(f, np.float64, 6 * Ndepth)
            _ = np.fromfile(f, np.int32, 1)

    return Atmost(0, 0, vturb=2e5 * np.ones(Ndepth), time=np.arange(Ntime, dtype=np.float64) * 0.1,
                  dt=np.ones(Ntime) * 0.1, z1=-z1t, d1=d1t, ne1=ne1t, tg1=tg1t, vz1=-vz1t, nh1=nh1t, bheat1=np.array(()))




