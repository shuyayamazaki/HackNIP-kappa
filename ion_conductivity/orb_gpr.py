# Gaussian process regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from jarvis.core.atoms import Atoms as JAtoms

import ase
from ase.build import bulk

from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs

device = "cpu"  # or device="cuda"
orbff = pretrained.orb_v2(device=device)
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)
graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device=device)

# Optionally, batch graphs for faster inference
# graph = batch_graphs([graph, graph, ...])

result = orbff.predict(graph)

# Convert to ASE atoms (unbatches the results and transfers to cpu if necessary)
atoms = atomic_system.atom_graphs_to_ase_atoms(
    graph,
    energy=result["graph_pred"],
    forces=result["node_pred"],
    stress=result["stress_pred"]
)
# ase atoms from cif
def cif2ase(cif):
    jatoms = JAtoms.from_cif(from_string=cif)
    atoms = jatoms.ase_converter()
    return atoms
def get_graph_features_from_orbff(atoms):
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device=device)
    graph = orbff.model.featurize_edges(graph)
    graph = orbff.model.featurize_nodes(graph)
    graph = orbff.model._encoder(graph)

    for gnn in orbff.model.gnn_stacks:
        graph = gnn(graph)
    
    return graph.node_features['feat'].mean(dim=0).numpy()
df = pd.read_parquet('/home/lucky/Projects/ion_conductivity/data/MPContribs_armorphous_diffusivity.parquet')
df['feat'] = df['cif'].apply(cif2ase).apply(get_graph_features_from_orbff).apply(list)
df['y'] = df['data_properties_A_diffusivity_value'].apply(np.log10)

X = np.array(df['feat'].to_list())
y = df['y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
# gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp = GaussianProcessRegressor(kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale = 1, nu = 1), n_restarts_optimizer = 30)
gp.fit(X_train, y_train)

y_pred, sigma = gp.predict(X_test, return_std=True)
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# Plot
import matplotlib.pyplot as plt
# Figure size
plt.figure(figsize=(6, 6))
# Font arial 12
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 12})  # Set default font size for all elements

plt.scatter(10**y_test, 10**y_pred)
y_max = max(10**y_test)

plt.plot([0, y_max], [0, y_max], color='red')
plt.xlabel('Ground truth')
plt.ylabel('Prediction')
plt.xlim(0, y_max)
plt.ylim(0, y_max)
# remove top and right border
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.savefig('orb_gpr.png', dpi=300)
plt.show()

# save GP model to disk
import pickle
with open('orb_gpr.pkl', 'wb') as f:
    pickle.dump(gp, f)


# # Plot residuals
# plt.scatter(y_test, y_pred-y_test)
# plt.axhline(0, color='red')
# plt.xlabel('True')
# plt.ylabel('Residual')
# plt.show()

