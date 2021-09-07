import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom, He_9_atom
from lightweaver.atmosphere import Atmosphere, ScaleType
from lightweaver.atomic_set import RadiativeSet, SpeciesStateTable
from lightweaver.molecule import MolecularTable
from lightweaver.LwCompiled import LwContext
from lightweaver.utils import InitialSolution
import lightweaver.constants as Const
from typing import List
from copy import deepcopy
from MsLightweaverAtoms import H_6, CaII, H_6_noLybb, H_6_nasa, CaII_nasa
import os
import os.path as path
import time
# from notify_run import Notify
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import multiprocessing
from pathlib import Path
from MsLightweaverManager import MsLightweaverManager
from MsLightweaverUtil import test_timesteps_in_dir, optional_load_starting_context, kill_child_processes
from ReadAtmost import read_atmost

OutputDir = 'TimestepsAdvNrLosses/'
Path(OutputDir).mkdir(parents=True, exist_ok=True)
Path(OutputDir + '/Rfs').mkdir(parents=True, exist_ok=True)
Path(OutputDir + '/ContFn').mkdir(parents=True, exist_ok=True)
NasaAtoms = [H_6_nasa(), CaII_nasa(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
             MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaNoLybbAtoms = [H_6_noLybb(), CaII(), He_9_atom(), C_atom(), O_atom(), 
                     Si_atom(), Fe_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaAtoms = [H_6(), CaII(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
                MgII_atom(), N_atom(), Na_atom(), S_atom()]
AtomSet = FchromaAtoms
ConserveCharge = False
PopulationTransportMode = 'Advect'
DetailedH = True
DetailedHPath = 'TimestepsAdvNrLosses'
Prd = False
MaxProcesses = -1
if MaxProcesses < 0:
    MaxProcesses = 10000

filesInOutDir = [f for f in os.listdir(OutputDir) if f.startswith('Step_')]
if len(filesInOutDir) > 0:
    print('Timesteps already present in output directory (%s), proceed? [Y/n]' % OutputDir)
    inp = input()
    if len(inp) > 0 and inp[0].lower() == 'n':
        raise ValueError('Data in output directory')

atmost = read_atmost('atmost.dat')
atmost.to_SI()
if atmost.bheat1.shape[0] == 0:
    atmost.bheat1 = np.load('BheatInterp.npy')

startingCtx = optional_load_starting_context(OutputDir)
# if startingCtx is None:
#     raise ValueError('No starting context found in %s' % OutputDir)

start = time.time()
ms = MsLightweaverManager(atmost=atmost, outputDir=OutputDir,
                          atoms=AtomSet,
                          activeAtoms=['H', 'Ca'], startingCtx=startingCtx,
                          detailedH=DetailedH, detailedHPath=DetailedHPath,
                          conserveCharge=ConserveCharge,
                          populationTransportMode=PopulationTransportMode,
                          prd=Prd)
ms.initial_stat_eq()
if ms.ctx.Nthreads > 1:
    ms.ctx.Nthreads = 1

# timeIdxs = np.linspace(0.5, 20, 40)
# timeIdxs = np.linspace(0.1, 40, 134)
# timeIdxs = np.array([2.0, 5.0, 10.0, 11.0, 15.0, 20.0, 30.0])
timeIdxs = np.array([11.0, 20.0])
pertSize = 50
pertSizeNePercent = 0.005
pertSizeVlos = 20
# dts = [5e-4, 1e-3]
dts = [1e-3, 1e-5, 1e-8]
dts = [1.0e10]
dts = [1.1e-2, 1.1e-3, 1.1e-4, 1.1e-6]
# dts = [1e3]

step = 545
Nspace = ms.atmos.height.shape[0]

def shush(fn, *args, **kwargs):
    with open(os.devnull, 'w') as f:
        with redirect_stdout(f):
            return fn(*args, **kwargs)

def rf_k(k, dt, Jstart=None):
    plus, minus = ms.rf_k(step, dt, pertSize, k, Jstart=Jstart)
    return plus, minus

def rf_k_se(k, dt, Jstart=None):
    plus, minus = ms.rf_k_stat_eq(step, dt, pertSize, k, Jstart=Jstart)
    return plus, minus

def rf_ne_k(k, dt, Jstart=None):
    plus, minus = ms.rf_ne_k(step, dt, pertSizeNePercent, k, Jstart=Jstart)
    return plus, minus

def rf_vlos_k(k, dt, Jstart=None):
    plus, minus = ms.rf_vlos_k(step, dt, pertSizeVlos, k, Jstart=Jstart)
    return plus, minus

if __name__ == '__main__':
    maxCpus = min(68, multiprocessing.cpu_count(), MaxProcesses)
    for t in timeIdxs:
        step = np.argwhere(np.abs(ms.atmost.time - t) < 1e-8).squeeze()

        contData = ms.cont_fn_data(step)
        with open(OutputDir + 'ContFn/ContFn_%d.pickle' % (step), 'wb') as pkl:
            pickle.dump(contData, pkl)

        for dt in dts:
            print('------- %d (%.2e s -> %.2e + %.2e s) -------' % (step, ms.atmost.time[step], ms.atmost.time[step], dt))

            print('Temperature')
            with ProcessPoolExecutor(max_workers=maxCpus) as exe:
                try:
                    futures = [exe.submit(shush, rf_k, k, dt, Jstart=contData['J']) for k in range(Nspace)]

                    for f in tqdm(as_completed(futures), total=len(futures)):
                        pass
                except KeyboardInterrupt:
                    exe.shutdown(wait=False)
                    kill_child_processes(os.getpid())


            rfPlus = np.zeros((Nspace, ms.ctx.spect.wavelength.shape[0]))
            rfMinus = np.zeros((Nspace, ms.ctx.spect.wavelength.shape[0]))

            for k, f in enumerate(futures):
                res = f.result()
                rfPlus[k, :] = res[0]
                rfMinus[k, :] = res[1]

            rf = (rfPlus - rfMinus) / pertSize
            with open(OutputDir + 'Rfs/Rf_temp_%.2e_%.2e_%d.pickle' % (pertSize, dt, step), 'wb') as pkl:
                pickle.dump({'rf': rf, 'pertSize': pertSize, 'dt': dt, 'step': step}, pkl)
            
            # print('ne')
            # with ProcessPoolExecutor(max_workers=maxCpus) as exe:
                # try:
                    # futures = [exe.submit(shush, rf_ne_k, k, dt, Jstart=contData['J']) for k in range(Nspace)]

                    # for f in tqdm(as_completed(futures), total=len(futures)):
                        # pass
                # except KeyboardInterrupt:
                    # exe.shutdown(wait=False)
                    # kill_child_processes(os.getpid())


            # rfPlus = np.zeros((Nspace, ms.ctx.spect.wavelength.shape[0]))
            # rfMinus = np.zeros((Nspace, ms.ctx.spect.wavelength.shape[0]))

            # for k, f in enumerate(futures):
                # res = f.result()
                # rfPlus[k, :] = res[0]
                # rfMinus[k, :] = res[1]

            # rf = (rfPlus - rfMinus) / pertSize
            # with open(OutputDir + 'Rfs/Rf_ne_%.2e_%.2e_%d.pickle' % (pertSizeNePercent, dt, step), 'wb') as pkl:
                # pickle.dump({'rf': rf, 'pertSizePercent': pertSizeNePercent, 'dt': dt, 'step': step}, pkl)
            
            # print('vlos')
            # with ProcessPoolExecutor(max_workers=maxCpus) as exe:
                # try:
                    # futures = [exe.submit(shush, rf_vlos_k, k, dt, Jstart=contData['J']) for k in range(Nspace)]

                    # for f in tqdm(as_completed(futures), total=len(futures)):
                        # pass
                # except KeyboardInterrupt:
                    # exe.shutdown(wait=False)
                    # kill_child_processes(os.getpid())


            # rfPlus = np.zeros((Nspace, ms.ctx.spect.wavelength.shape[0]))
            # rfMinus = np.zeros((Nspace, ms.ctx.spect.wavelength.shape[0]))

            # for k, f in enumerate(futures):
                # res = f.result()
                # rfPlus[k, :] = res[0]
                # rfMinus[k, :] = res[1]

            # rf = (rfPlus - rfMinus) / pertSize
            # with open(OutputDir + 'Rfs/Rf_vlos_%.2e_%.2e_%d.pickle' % (pertSizeVlos, dt, step), 'wb') as pkl:
                # pickle.dump({'rf': rf, 'pertSize': pertSizeVlos, 'dt': dt, 'step': step}, pkl)
