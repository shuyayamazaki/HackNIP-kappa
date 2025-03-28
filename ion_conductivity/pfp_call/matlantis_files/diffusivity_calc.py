import argparse
import pathlib
import numpy as np
import pandas as pd
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase import units
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from time import perf_counter
from scipy import stats

# Set up constants and directories
EXAMPLE_DIR = pathlib.Path("/home/jovyan/SKim/Diffusivity/MD_Li_diffusion_in_LGPS/").resolve()
INPUT_DIR = EXAMPLE_DIR / "input"
OUTPUT_DIR = EXAMPLE_DIR / "output"

# Initialize estimator and calculator
estimator = Estimator(model_version="v2.0.0", calc_mode=EstimatorCalcMode.CRYSTAL)
calculator = ASECalculator(estimator)

def run_md_simulation(temperature, index):
    """
    Run MD simulation at the specified temperature.

    Args:
        temperature (float): Temperature in Kelvin for the simulation.

    Returns:
        float: Diffusion coefficient (D) in cm^2/sec (0.0001 * A^2/psec).
    """
    s_time = perf_counter()

    # Load optimized structure
    structure = read(INPUT_DIR / f"temp_{index}.cif")
    structure.calc = calculator

    # Initialize velocity distribution
    rng = np.random.default_rng(seed=42)
    MaxwellBoltzmannDistribution(structure, temperature_K=temperature, rng=rng)

    # Set up MD simulation
    traj_file = OUTPUT_DIR / f"MD_{index}.traj" # _{int(temperature):04}
    log_file = OUTPUT_DIR / f"MD_{index}.log" # _{int(temperature):04}

    if traj_file.exists():
        traj_file.unlink()
        print(f"[INFO] Removed existing trajectory file: {traj_file}")
    if log_file.exists():
        log_file.unlink()
        print(f"[INFO] Removed existing log file: {log_file}")

    dyn = Langevin(
        structure,
        2 * units.fs,  # Time step in femtoseconds
        temperature_K=temperature,
        friction=0.02,
        trajectory=str(traj_file),
        loginterval=100,
        rng=rng
    )
    dyn.attach(MDLogger(dyn, structure, str(log_file), header=False, stress=False, peratom=True, mode="w"), interval=100)

    # Run MD for 1250 steps
    dyn.run(1250)
    proctime = perf_counter() - s_time

    # Analyze trajectory for diffusion coefficient
    trajectory = read(traj_file, index="::")
    positions_all = np.array([trajectory[i].get_positions() for i in range(len(trajectory))])
    Li_index = [i for i, x in enumerate(trajectory[0].get_chemical_symbols()) if x == 'Li']
    positions = positions_all[:, Li_index]
    msd = np.mean(np.sum((positions - positions[0]) ** 2, axis=2), axis=1)

    slope, _, _, _, _ = stats.linregress(range(len(msd)), msd)
    D = slope / 60000

    print(f"Diffusion coefficient {D:.10f} cm^2/sec")
    return D

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MD simulation to calculate diffusion coefficient.")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature (in K) for the MD simulation.")
    parser.add_argument("--index", type=int, required=True, help="Unique index for distinguishing this simulation.")
    args = parser.parse_args()

    # Run MD simulation and calculate diffusion coefficient
    D = run_md_simulation(args.temperature, args.index)

    print(D)
