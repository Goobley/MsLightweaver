import numpy as np
from scipy.interpolate import RectBivariateSpline
from lightweaver import constants as Const

# TODO(cmo): This should really be done with a generator/coroutine
def get_next_line(data):
    if len(data) == 0:
        return None
    for i, d in enumerate(data):
        if d.strip().startswith('*') or d.strip() == '':
            continue
        break
    d = data[i]
    if i == len(data) - 1:
        data[:] = []
        return d.strip()
    data[:] = data[i+1:]
    return d.strip()

class EmisTable:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            data = f.readlines()

        Ntemp = int(get_next_line(data))
        Nlambda = int(get_next_line(data))
        self.Ntemp = Ntemp
        self.Nlambda = Nlambda
        self.binCentres = np.fromstring(get_next_line(data), sep=' ')
        self.binWidth = np.fromstring(get_next_line(data), sep=' ')

        logT = []
        logEmis = []
        for t in range(Ntemp):
            line = get_next_line(data)
            split = line.split()
            logT.append(float(split[0]))
            emis = [float(l) for l in split[1:]]
            logEmis.append(emis)

        logT = np.array(logT)
        self.logT = logT
        logEmis = np.array(logEmis)
        self.logEmis = logEmis
        self.emisInterp = RectBivariateSpline(logT, self.binCentres, logEmis, kx=1, ky=1)

    def compute_downgoing_radiation(self, wvl, atmos):
        totEmis = np.zeros(wvl.shape[0])
        wvlMask = wvl >= 8000
        neCm3 = atmos.ne / 1e6
        nhCm3 = atmos.nHTot / 1e6
        z = atmos.z
        dz = np.zeros_like(z)
        dz[1:] = z[1:] - z[:-1]
        dz[0] = dz[1]
        dz *= 1e2
        wvlAngstrom = wvl * 10
        temp = atmos.temperature

        for k in range(temp.shape[0]):
            if temp[k] < 7e4:
                continue

            emis = self.emisInterp(np.log10(temp[k]), wvlAngstrom).squeeze()
            totEmis += 10**emis * neCm3[k] * nhCm3[k] * (-dz[k]) / 2.0 / np.pi * wvlAngstrom**2 / (Const.CLight*1e2) / 1e8
            # NOTE(cmo): Should the factor not be 4pi for energy conservation?
            # totEmis += 10**emis * neCm3[k] * nhCm3[k] * (-dz[k]) * wvlAngstrom**2 / (Const.CLight*1e2) / 1e8

        totEmis[wvlMask] = 0
        # NOTE(cmo): This is still in CGS, should be erg/s/cm2/Hz, convert to J/s/m2/Hz
        totEmis /= 1e3

        result = totEmis[:, None] / atmos.muz[None, :]

        return result
