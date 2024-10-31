# %% [markdown]
# Copyright Preferred Computational Chemistry, Inc. as contributors to Matlantis contrib project

import pathlib
EXAMPLE_DIR = pathlib.Path("/home/jovyan/SKim/Diffusivity/MD_Li_diffusion_in_LGPS/").resolve()
INPUT_DIR = EXAMPLE_DIR / "input"
OUTPUT_DIR = EXAMPLE_DIR / "output_2"
OUTPUT_DIR.mkdir(exist_ok=True)

# %%
print(pathlib.Path("__file__").resolve())

# %%
# common modules
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import PillowWriter
import seaborn as sns
import math
import optuna
import os,sys,csv,glob,shutil,re,time
from time import perf_counter
from joblib import Parallel, delayed

from sklearn.metrics import mean_absolute_error

import ase
from ase.visualize import view
from ase.optimize import BFGS
from ase.constraints import FixAtoms, FixedPlane, FixBondLength, ExpCellFilter
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase import Atoms
from ase.io import read, write
from ase.io import Trajectory
from ase import units
from my_utils.optimization_utils import myopt, opt_cell_size

from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode

estimator = Estimator(model_version="v2.0.0", calc_mode=EstimatorCalcMode.CRYSTAL)
calculator = ASECalculator(estimator)

# %%
def myopt(m,sn = 10,constraintatoms=[],cbonds=[]):
    fa = FixAtoms(indices=constraintatoms)
    fb = FixBondLengths(cbonds,tolerance=1e-5,)
    m.set_constraint([fa,fb])
    m.set_calculator(calculator)
    maxf = np.sqrt(((m.get_forces())**2).sum(axis=1).max())
    print("ini   pot:{:.4f},maxforce:{:.4f}".format(m.get_potential_energy(),maxf))
    de = -1 
    s = 1
    ita = 50
    while ( de  < -0.001 or de > 0.001 ) and s <= sn :
        opt = BFGS(m,maxstep=0.04*(0.9**s),logfile=None)
        old  =  m.get_potential_energy() 
        opt.run(fmax=0.0005,steps =ita)
        maxf = np.sqrt(((m.get_forces())**2).sum(axis=1).max())
        de =  m.get_potential_energy()  - old
        print("{} pot:{:.4f},maxforce:{:.4f},delta:{:.4f}".format(s*ita,m.get_potential_energy(),maxf,de))
        s += 1
    return m

def opt_cell_size(m,sn = 10, iter_count = False): # m:Atoms object
    m.set_constraint() # clear constraint
    m.set_calculator(calculator)
    maxf = np.sqrt(((m.get_forces())**2).sum(axis=1).max()) # get max value of √(fx^2 + fy^2 + fz^2)
    ucf = ExpCellFilter(m)
    print("ini   pot:{:.4f},maxforce:{:.4f}".format(m.get_potential_energy(),maxf))
    de = -1 
    s = 1
    ita = 50
    while ( de  < -0.01 or de > 0.01 ) and s <= sn :
        opt = BFGS(ucf,maxstep=0.04*(0.9**s),logfile=None)
        old  =  m.get_potential_energy() 
        opt.run(fmax=0.005,steps =ita)
        maxf = np.sqrt(((m.get_forces())**2).sum(axis=1).max())
        de =  m.get_potential_energy()  - old
        print("{} pot:{:.4f},maxforce:{:.4f},delta:{:.4f}".format(s*ita,m.get_potential_energy(),maxf,de))
        s += 1
    if iter_count == True:
        return m, s*ita
    else:
        return m
    

# %% [markdown]

# %%
bulk = read(INPUT_DIR / "Li10Ge(PS6)2_mp-696138_computed.cif")
bulk.calc = calculator

print("# atoms =", len(bulk))
print("Initial lattice constant =", bulk.cell.cellpar())

opt_cell_size(bulk)
print ("Optimized lattice constant =", bulk.cell.cellpar())

# %%
optimized_cellpar = np.array([float(f'{x:.3f}') for x in bulk.cell.cellpar()])
bulk.set_cell(optimized_cellpar, scale_atoms=True)

# %%
# Remove comment out below if you want to run MD with bigger systems.
bulk = bulk.repeat([2,2,1])

# %%
print ("Optimized lattice constant =", bulk.cell.cellpar())

# %%
os.makedirs(OUTPUT_DIR / "structure/", exist_ok=True)
write(OUTPUT_DIR / "structure/opt_structure.xyz", bulk)

# %%
with open(OUTPUT_DIR / "structure/opt_structure.xyz", 'w') as f:
    symbols = bulk.get_chemical_symbols()
    positions = bulk.get_positions()
    f.write(f'{len(bulk)}\n')
    f.write('Truncated to 4 decimal places\n')
    for s, p in zip(symbols, positions):
        f.write(f'{s} {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n')

# %%
with open(OUTPUT_DIR / "structure/opt_structure.xyz", "r") as f:

    cif_content = f.read()
    # print(cif_content)

# %% [markdown]
# Check number of Li in this structure

# %%
Li_index = [i for i, x in enumerate(bulk.get_chemical_symbols()) if x == 'Li']
print(len(Li_index))

# %% [markdown]
# ## Running MD simulation with various temperature
# 
# Here ASE's `Langevin` class is used for MD simulation.<br/>
# Some kinds of NVT simulation are supported in ASE, please refer detail below: 
#  - https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md.langevin
# 
# We run MD with various temperature configuration to see the Arrhenius plot later.<br/>
# MD is parallelized using `joblib` module.

# %%
temp_list = [723, 923, 1023]
# temp_list = [423, 523, 623, 723, 823, 923, 973, 1023]

# %%
os.makedirs(OUTPUT_DIR / "traj_and_log/", exist_ok=True)

def run_md(i):
    s_time = perf_counter()
    
    estimator = Estimator(model_version="v2.0.0", calc_mode=EstimatorCalcMode.CRYSTAL)
    calculator = ASECalculator(estimator)
    
    t_step = 2     # as fs
    temp = i       # as K
    itrvl = 100
    rng = np.random.default_rng(seed=42)
    
    structure = read(OUTPUT_DIR/ "structure/opt_structure.xyz")
    structure.calc = calculator
    
    MaxwellBoltzmannDistribution(structure, temperature_K=temp, rng = rng)

    dyn = Langevin(
        structure,
        t_step * units.fs,
        temperature_K=temp,
        friction=0.02,
        trajectory=f"{OUTPUT_DIR}/traj_and_log/MD_{str(i).zfill(4)}.traj",
        loginterval=itrvl,
        append_trajectory=False,
        rng = rng
    )
    dyn.attach(MDLogger(dyn, structure, f"{OUTPUT_DIR}/traj_and_log/MD_{str(i).zfill(4)}.log", header=False, stress=False,
               peratom=True, mode="w"), interval=itrvl)

    dyn.run(1250)
    # dyn.run(2_000_000)
    proctime = perf_counter() - s_time

    return([i, proctime/3600])

# %%
results = Parallel(n_jobs=len(temp_list), verbose=1)(delayed(run_md)(i) for i in temp_list)

# %%


# %% [markdown]
# Copyright Preferred Computational Chemistry, Inc. as contributors to Matlantis contrib project
# 
# # MD Li diffusion in LGPS - postprocess analysis
# 
# We analyze the MD result of LPGS Li diffusion in this script.

# %% [markdown]
# ## Setup

# %%
# Please install these libraries only for first time of execution
# !pip install pandas matplotlib scipy ase

# %%
import pathlib

# %%
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import glob, os

import ase
from ase.visualize import view
from ase.io import read, write
from ase.io import Trajectory

# %% [markdown]
# ## Analyze specific temperature
# 
# We read trajectory file with 20 step interval.
# 
# Since trajectory file is saved with 0.05 ps interval in the previous MD simulation, this `trj` is 1ps interval.

# %%
temp = 1023

trj = read(OUTPUT_DIR / f"traj_and_log/MD_{temp:04}.traj", index="::")
# view(trj, viewer = "ngl")

# %%
Li_index = [i for i, x in enumerate(trj[0].get_chemical_symbols()) if x == 'Li']
print(len(Li_index))

# %% [markdown]
# To calculate diffusion coefficient, we can calculate mean square displacement, variance of positions.

# %%
# t0 = len(trj) // 2
t0 = 0

positions_all = np.array([trj[i].get_positions() for i in range(t0, len(trj))])

# shape is (n_traj, n_atoms, 3 (xyz))
print("positions_all.shape: ", positions_all.shape)

# position of Li 
positions = positions_all[:, Li_index]
positions_x = positions[:, :, 0]
positions_y = positions[:, :, 1]
positions_z = positions[:, :, 2]

print("positions.shape    : ", positions.shape)
print("positions_x.shape  : ", positions_x.shape)

# %%
# msd for each x,y,z axis
msd_x = np.mean((positions_x-positions_x[0])**2, axis=1)
msd_y = np.mean((positions_y-positions_y[0])**2, axis=1)
msd_z = np.mean((positions_z-positions_z[0])**2, axis=1)

# total msd. sum along xyz axis & mean along Li atoms axis.
msd = np.mean(np.sum((positions-positions[0])**2, axis=2), axis=1)

# %% [markdown]
# At first, we can see from the figure that diffusion is prominent along z-axis than x, y-axis, which matches the known fact.

# %% [markdown]
# Next, we can fit line to the total msd to obtain diffusion coefficient.

# %%

# %% [markdown]
# Here `scipy.stats.linregress` is used to fit straight line.
#  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
#  
# The coefficient 6 comes from the degree of freedom in xyz-axis.

# %%
slope, intercept, r_value, _, _ = stats.linregress(range(len(msd)), msd)
D = slope / 6
print(slope, intercept, r_value)
print(f"Diffusion coefficient {D:.2f} A^2/psec")

# %%
t = np.arange(len(msd))

# %% [markdown]
# Convert diffusion coefficient from A^2/ps to cm^2/sec unit, take log.

# %%
np.log10(D*1e-16*1e12)

# %% [markdown]
# ## Analyze temperature dependency
# 
# We can calculate activation energy $E_A$ from Arrhenius equation
#  
# $$D = D_0 \exp \left(- \frac{E_A}{RT} \right)$$
# 
# where $D$ is the diffusion coefficient at temperature $T$.
# 
#  - https://en.wikipedia.org/wiki/Arrhenius_equation
# 
# By taking log to each side, we obtain
# 
# $$\log D = \log D_0 - \frac{E_A}{RT}.$$
# 
# Thus when we plot $\log D$ as y-axis and $1/T$ as x-axis, we can calculate activation energy $E_A$ from the slope value.

# %% [markdown]
# We calculate diffusion coefficient for each temperature by following the same procedure in previous section.

# %%
trj_list = sorted(glob.glob(f"{OUTPUT_DIR}/traj_and_log/*.traj"))

# %%
trj_list

# %%
t0 = 0

os.makedirs(OUTPUT_DIR / "msd/", exist_ok=True)
D_list = []
for path in trj_list:
    trj = read(path, index="::1")
    Li_index = [Li_i for Li_i, x in enumerate(trj[0].get_chemical_symbols()) if x == 'Li']

    # msd for each x,y,z axis
    positions_all = np.array([trj[i].get_positions() for i in range(t0, len(trj))])
    positions = positions_all[:, Li_index]
    positions_x = positions[:, :, 0]
    positions_y = positions[:, :, 1]
    positions_z = positions[:, :, 2]
    # msd for each x,y,z axis
    msd_x = np.mean((positions_x-positions_x[0])**2, axis=1)
    msd_y = np.mean((positions_y-positions_y[0])**2, axis=1)
    msd_z = np.mean((positions_z-positions_z[0])**2, axis=1)

    # total msd. sum along xyz axis & mean along Li atoms axis.
    msd = np.mean(np.sum((positions-positions[0])**2, axis=2), axis=1)
    
    slope, intercept, r_value, _, _ = stats.linregress(range(len(msd)), msd)
    logD = np.log10(slope*1e-16*1e12/6)
    T = int(os.path.basename(path.split(".")[0].replace("MD_","").replace("traj_and_log/","")))
    D_list.append([T, 1000/T, logD])
    
# %%
df = pd.DataFrame(D_list, columns=["T", "1000/T", "logD"])

# %%
df

# %%
sl, ic, rv, _, _ = stats.linregress(df["1000/T"], df["logD"])
print(sl, ic, rv)

# %% [markdown]
# Arrhenius plot, following the reference
#  - [First Principles Study of the Li10GeP2S12 Lithium Super Ionic Conductor Material](https://pubs.acs.org/doi/10.1021/cm203303y)

# %% [markdown]
# Finally we can calculate activation energy from the arrhenius plot.
# 
# Below, the term `1000 * np.log(10)` is to fix the x, y-axis scale.

# %%
from ase.units import J, mol
R = 8.31446261815324  # J/(K・mol)


E_act = -sl * 1000 * np.log(10) * R * (J / mol)
print(f"Activation energy: {E_act* 1000:.1f} meV")