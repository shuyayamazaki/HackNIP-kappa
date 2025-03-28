# %%
!pip install matplotlib ase rdfpy

# %%
# common modules
import numpy as np
import matplotlib.pyplot as plt
from rdfpy import rdf
from ase.io import read
from ase.build import make_supercell
from pymatgen.core import Structure

# %%
def cal_avg_rdf(trajectory, dr):
    """Calculates the average PDF and standard deviation from a trajectory.

    Args:
        trajectory: A list of ASE Atoms objects representing the snapshots.
        r_max: Maximum distance for the PDF calculation.
        dr: Distance bin size.

    Returns:
        A tuple of r values, average PDF, and standard deviation of the PDF.
    """

    P = [[2, 0, 0],
         [0, 2, 0],
         [0, 0, 2]]
    g_rs = []
    max_length = 0
    max_r = None  # Initialize max_r to None

    for snapshot in trajectory:
        supercell = make_supercell(snapshot, P)
        coords = supercell.positions
        g_r, radii = rdf(coords, dr)
        g_rs.append(g_r)
        max_length = max(max_length, len(g_r))

        # Update max_r if a longer g_r is found
        if len(g_r) == max_length:
            max_r = radii

    # Pad shorter arrays with zeros
    for i in range(len(g_rs)):
        if len(g_rs[i]) < max_length:
            g_rs[i] = np.pad(g_rs[i], (0, max_length - len(g_rs[i])), 'constant')

    g_rs = np.array(g_rs)
    avg_g_r = np.mean(g_rs, axis=0)
    std_g_r = np.std(g_rs, axis=0)

    # Use max_r if it's not None, otherwise use the original r calculation
    if max_r is not None:
        r = max_r
    else:
        r = np.arange(0, len(avg_g_r)*dr, dr)

    return r, avg_g_r, std_g_r

# %% Rdf for a crystalline structure
structure_Ti = Structure.from_file(f"output_100/Ti_mp-46_computed.cif")
structure_Ti.make_supercell(9)

coords_Ti = structure_Ti.cart_coords
noise_Ti = np.random.normal(loc=0.0, scale=0.05, size=(coords_Ti.shape))
coords_Ti = coords_Ti + noise_Ti

g_r_Ti, radii_Ti = rdf(coords_Ti, dr=0.05)
plt.plot(radii_Ti, g_r_Ti)

# %% Statistical rdf for an amorphous structure
traj_ref_pfp = read(f"output_100/traj_and_log/MD_ref.traj", ":")

dr = 0.15
r, avg_rdf, std_rdf = cal_avg_rdf(traj_ref_pfp, dr)

plt.errorbar(r, avg_rdf, yerr=std_rdf)
plt.xlabel('r (Å)')
plt.ylabel('g(r)')
plt.show()


