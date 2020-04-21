import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom, CaII_atom
import lightweaver as lw
from MsLightweaverAtoms import H_6, CaII, He_9, H_6_nasa, CaII_nasa
from pathlib import Path
import os
import os.path as path
import time
from notify_run import Notify
from MsLightweaverManager import MsLightweaverManager, optional_load_starting_context
from MsLightweaverUtil import test_timesteps_in_dir

OutputDir = 'TimestepsHeightWiderTest1000/'
Path(OutputDir).mkdir(parents=True, exist_ok=True)
Path(OutputDir + '/Rfs').mkdir(parents=True, exist_ok=True)
Path(OutputDir + '/ContFn').mkdir(parents=True, exist_ok=True)
NumInterfaces = 1024
NasaAtoms = [H_6_nasa(), CaII_nasa(), He_9(), C_atom(), O_atom(), Si_atom(), Fe_atom(), 
             MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaAtoms = [H_6(), CaII(), He_9(), C_atom(), O_atom(), Si_atom(), Fe_atom(), 
                MgII_atom(), N_atom(), Na_atom(), S_atom()]
DoAdvection = False

test_timesteps_in_dir(OutputDir) 

with open('RadynData.pickle', 'rb') as pkl:
    atmost = pickle.load(pkl)

startingCtx = optional_load_starting_context(OutputDir)

start = time.time()
ms = MsLightweaverManager(atmost=atmost, outputDir=OutputDir, 
                          numInterfaces=NumInterfaces, atoms=FchromaAtoms, 
                          activeAtoms=['H', 'Ca'], startingCtx=startingCtx,
                          doAdvection=DoAdvection)
ms.initial_stat_eq(popTol=1e-3)
ms.save_timestep()

if startingCtx is None:
    with open(OutputDir + 'StartingContext.pickle', 'wb') as pkl:
        pickle.dump(ms.ctx, pkl)

maxSteps = ms.atmost['time'].shape[0] - 1
ms.atmos.bHeat[:] = ms.atmost['bheat1'][0]
firstStep = 0
# firstStep = 7685
if firstStep != 0:
    ms.load_timestep(firstStep)
    ms.ctx.spect.J[:] = 0.0
    ms.ctx.formal_sol_gamma_matrices()

for i in range(firstStep, maxSteps):
    stepStart = time.time()
    if i != 0:
        ms.increment_step()
    ms.time_dep_step(popsTol=1e-3, JTol=5e-3, nSubSteps=1000, theta=0.55)
    ms.ctx.clear_ng()
    ms.save_timestep()
    stepEnd = time.time()
    print('-------')
    print('Timestep %d done (%f s)' % ((i+1), ms.atmost['time'][i+1]))
    print('Time taken for step %.2e s' % (stepEnd - stepStart))
    print('-------')
end = time.time()
print('Total time taken %.4e' % (end - start))

notify = Notify()
notify.read_config()
notify.send('MsLightweaver done!')