# import numpy as np
import requests
import pandas as pd
import numpy as np
# from jarvis.core.atoms import Atoms
# from jarvis.core.graphs import Graph
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from pathlib import Path
from typing import Union

from typing_extensions import TypeAlias

PathType: TypeAlias = Union[str, Path]
StringOrNumber: TypeAlias = Union[str, int, float]

curr_dir = os.path.dirname(os.path.abspath(__file__))
import os
import sys
import time
from typing import List

# from loguru import logger

# from settings import BASE_OUTDIR

__all__ = ["enable_logging", "make_outdir"]
from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_molecular_weight(smiles):
    """
    Calculate the molecular weight of a molecule given its SMILES representation.

    Parameters:
    smiles (str): The SMILES string representing the molecule.

    Returns:
    float: The molecular weight of the molecule.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string provided.")
    return Descriptors.MolWt(molecule)


def make_outdir(run_name):
    """Make a directory if the current date and time in the format YYYYMMDD_HHMMSS.

    If run_name is specified, append it to the directory name in the format YYYYMMDD_HHMMSS_run_name.
    """
    outdir = os.path.abspath(os.path.join(BASE_OUTDIR, time.strftime("%Y%m%d_%H%M%S")))
    if run_name is not None:
        outdir = f"{outdir}_{run_name}"
    os.makedirs(outdir, exist_ok=True)
    return outdir


def enable_logging() -> List[int]:
    """Set up the gptchem logging with sane defaults."""
    logger.enable("gptchem")

    config = dict(
        handlers=[
            dict(
                sink=sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z UTC}</>"
                " <red>|</> <lvl>{level}</> <red>|</> <cyan>{name}:{function}:{line}</>"
                " <red>|</> <lvl>{message}</>",
                level="INFO",
            ),
            dict(
                sink=sys.stderr,
                format="<red>{time:YYYY-MM-DD HH:mm:ss.SSS Z UTC} | {level} | {name}:{function}:{line} | {message}</>",
                level="WARNING",
            ),
        ]
    )
    return logger.configure(**config)

def pubchem_request(chemical_name, key='name', task=None, max_records=1, structure_type=None):
    '''
    If task is None, return just result of compound name 
    
    input: chemical_name, key, task, max_records, structure_type
        chemical_name: str
        key: str
        task: str
        max_records: int
        structure_type: str
    output: list of dict
    '''
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/'
    if task == 'fastsimilarity_2d':
        url += task + '/'
    url += f'{key}/{chemical_name}/property/CanonicalSMILES,InChI,IUPACName/JSON?MaxRecords={max_records}'
    if structure_type == '3d':
        url += '&record_type=3d'
    response = requests.get(url)
    return response.json()['PropertyTable']['Properties']


def pubchem_request_synonym(chemical_name):
    response = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{chemical_name}/synonyms/JSON')
    return response.json()['InformationList']['Information'][0]['Synonym']


def search_mat_in_paper_return_paragraphs(paper_df, mat, paper_max=10):
    '''
    one to many function. one material to many papers
    input: paper_df, mat
        paper_df: dataframe of papers
        mat: material name
        paper_max: max number of papers to search
    output: list of paragraphs, list of dois
        paragraphs: list of paragraphs containing the material name
        dois: list of dois of the papers containing the material name
    '''
    # First check if the formula is in the text
    paper_df['query'] = paper_df['text'].apply(lambda x: mat in x)

    # select papers max number of papers to search
    true_rows = paper_df[paper_df['query']]
    selected_rows = true_rows if len(true_rows) <= paper_max else true_rows.sample(n=paper_max) 

    # return paragraphs and dois
    text_series = selected_rows['text'].apply(lambda x: [p for p in x.split('\n') if mat in p]).to_list()  # list of paragraph including keyword
    doi_series = selected_rows['doi'].to_list()
    return text_series, doi_series


def jarvis_atoms2graph(input_item):
    '''
    input: pandas dataframe rows
    output: graph in dict format

    Convert atoms column in parquet file to graphs
    Graphs contain
        edge_index: list of two lists of int
        edge_attr: list of float
        num_nodes: int
        node_feat: list of list of float
        y: list of int
    '''
    if 'atoms' in input_item.keys():
        jatoms = Atoms.from_dict(input_item['atoms'])
    elif 'cif' in input_item.keys():
        with open('temp.cif', 'w') as f:
            f.write(input_item['cif'])
        jatoms = Atoms.from_cif('temp.cif')
    dglgraph = Graph.atom_dgl_multigraph(jatoms, compute_line_graph=False, atom_features='cgcnn')
    item = {}
    edges = dglgraph.edges()
    item['edge_index'] = [edges[0].tolist(), edges[1].tolist()]
    item['edge_attr'] = dglgraph.edata['r'].tolist()
    item['node_feat'] = dglgraph.ndata['atom_features'].tolist()
    item['y'] = [0]
    return item


def json_atoms_to_graphs(json_file):
    '''
    Convert atoms column in json file to graphs
    Graphs contain
        edge_index: list of two lists of int
        edge_attr: list of float
        num_nodes: int
        node_feat: list of list of float
        y: list of int
    '''
    # Load parquet file
    df = pd.read_json(json_file)
    # Divide parquet file into chunks
    chunks = np.array_split(df, 12)
    def json_atoms_to_graphs_chunk(chunk):
        '''
        input: chunk of parquet file dataframe
        output: list of graphs
        '''
        result = []
        for atom in tqdm(chunk.iloc):
            item = jarvis_atoms2graph(atom)
            result.append(item)
        return result
    # Parallelize
    results = Parallel(n_jobs=12)(delayed(json_atoms_to_graphs_chunk)(chunk) for chunk in chunks)
    results = sum(results, [])
    return results    


def converting(key, query, ri):
    '''
    Args:
        key: the key of the column (desired column to get)
        query: the query of the row (desired reagent to get)
        ri: reagent information dataframe
    Returns:
        the value of the key in the row of the query. Desired property of the reagent.
    '''
    if ri is None:
        ri = pd.read_excel(os.path.join(curr_dir, 'Electrolyte dataset.xlsx'), sheet_name='Reagent information')
    return ri[ri['Representative Abbreviation'] == query][key].values[0]


def parse_solvent(input_string, chem_dict):
    '''
    example of input_string = "DME:TTE=1:1 (vol)"    
    Args:
        input_string: str
        chem_dict: dict
    Returns:
        list of tuples (solvent abbreviation, solvent SMILES, ratio)
    '''
    if '\n' in input_string:
        input_strings = input_string.split('\n')
        return sum([parse_solvent(input_string, chem_dict) for input_string in input_strings], [])
    elif '=' in input_string:
        input_string = input_string.split('=')
        abbs = input_string[0].split(':')
        mols = [chem_dict[abb.replace(' ', '')] for abb in abbs]
        ratios = input_string[1].split(' ')[0].split(':')
        ratios = [float(ratio) for ratio in ratios]
        return list(zip(abbs, mols, ratios))
    else:
        return [(input_string, chem_dict[input_string].replace(' ', ''), 1.0)] 
    

def parse_salt(input_string, chem_dict):
    '''
    example of input_string = "0.75 M LiFSI + 0.25 M LiTFSI"    
    Args:
        input_string: str
        chem_dict: dict
    Returns:
        list of tuples (salt abbreviation, salt SMILES, ratio, concentration type)
    '''
    # if '+' in input_string:
    if 'mol' not in input_string:
        input_string = input_string.split(' + ')  # ("0.75 M LiFSI", "0.25 M LiTFSI")
        abbs = [item.split(' ')[-1] for item in input_string]  # ("LiFSI", "LiTFSI")
        mols = [chem_dict[abb.replace(' ', '')] for abb in abbs]  
        ratios = [float(item.split(' ')[0]) for item in input_string]  # (0.75, 0.25)
        concen_type = [item.split(' ')[1] for item in input_string]  # ("M", "M")
        return list(zip(abbs, mols, ratios, concen_type))
    else:
        input_string = input_string.split(' + ')  # ("LiFSI (1 mol)", "LiTFSI (2 mol)")
        abbs = [item.split(' ')[0] for item in input_string]  # ("LiFSI", "LiTFSI")
        mols = [chem_dict[abb.replace(' ', '')] for abb in abbs]  
        ratios = [float(item.split(' ')[1].replace('(', '')) for item in input_string]  # (1, 2)
        concen_type = ['mol' for item in input_string]  # ("mol", "mol")
        return list(zip(abbs, mols, ratios, concen_type))
    # else:
    #     parsed = input_string.split(' ')
    #     return [(parsed[-1], chem_dict[parsed[-1].replace(' ', '')], float(parsed[0]), parsed[1])]  # (LiFSI, LiFSI, 0.75, M)
    
    
def parse_additive(input_string, chem_dict):
    '''
    example of input_string = "FEC (1 wt%) + VC (2 wt%) + TFEC (3 wt%)"    
    Args:
        input_string: str
        chem_dict: dict
    Returns:
        list of tuples (additive abbreviation, additive SMILES, ratio, concentration type)    
    '''
    if isinstance(input_string, float) or input_string == '/':
        return []
    elif '+' in input_string:
        input_string = input_string.split(' + ')  # ("FEC (1 wt%)", "VC (2 wt%)", "TFEC (3 wt%)")
        abbs = [item.split(' ')[0] for item in input_string]  # ("FEC", "VC", "TFEC")
        mols = [chem_dict[abb.replace(' ', '')] for abb in abbs]  
        ratios = [float(item.split(' ')[1].replace('(', '')) for item in input_string]  # (1, 2, 3)
        concen_type = [item.split(' ')[2].replace('%)', '%') for item in input_string]  # ("wt%", "wt%", "wt%")
        return list(zip(abbs, mols, ratios, concen_type))
    else:
        parsed = input_string.replace(' (', ' ').replace('%)', '%').split(' ')
        return [(parsed[0], chem_dict[parsed[0].replace(' ', '')], float(parsed[1]), parsed[-1])]  # (FEC, FEC, 1, wt%)
    

def parse_experiment(solvent, salt, additive, chem_dict, ri):
    '''
    1. Get solvent weight, volume, and mol
    2. Get Salt weight, volume, and mol
    3. Get Additive weight, volume, and mol

    '''
    # solvent = solvent.replace('\n', '')
    salt = salt.replace('\n', '')
    additive = additive if isinstance(additive, float) else additive.replace('\n', '')
    # Solvent
    solvent_parsed = parse_solvent(solvent, chem_dict)
    if '(vol)' in solvent or '(vt)' in solvent:  # ratio is volume
        solvent_volume = [ratio for abb, _, ratio in solvent_parsed]  # cm3
        solvent_weight = [converting('Density\n(g cm-3)', abb, ri) * ratio for abb, _, ratio in solvent_parsed]  # g
        solvent_mol = [converting('Density\n(g cm-3)', abb, ri) / calculate_molecular_weight(smi) * ratio for abb, smi, ratio in solvent_parsed]  # mol
    elif '(wt)' in solvent or 'mass ratio' in solvent:  # ratio is weight
        solvent_weight = [ratio for abb, _, ratio in solvent_parsed]  # g
        solvent_volume = [ratio / converting('Density\n(g cm-3)', abb, ri) for abb, _, ratio in solvent_parsed]  # cm3
        solvent_mol = [ratio / calculate_molecular_weight(smi) for abb, smi, ratio in solvent_parsed]  # mol
    elif '(molar ratio)' in solvent or '(mol)' in solvent:  # ratio is molar
        solvent_mol = [ratio for abb, _, ratio in solvent_parsed]  # mol
        solvent_weight = [calculate_molecular_weight(smi) * ratio for abb, smi, ratio in solvent_parsed]  # g = g/mol * mol
        solvent_volume = [weight / converting('Density\n(g cm-3)', abb, ri) for weight, (abb, _, ratio) in zip(solvent_weight, solvent_parsed)]  # cm3 = g / g/cm3
    else:  # only one solvent
        solvent_mol = [1]
        solvent_weight = [calculate_molecular_weight(solvent_parsed[0][1])]
        solvent_volume = [weight / converting('Density\n(g cm-3)', abb, ri) for weight, (abb, _, ratio) in zip(solvent_weight, solvent_parsed)]  # cm3 = g / g/cm3

    # Salt
    salt_parsed = parse_salt(salt, chem_dict)
    salt_mol = []
    salt_weight = []
    salt_volume = []
    for abb, smi, ratio, c_type in salt_parsed:
        if c_type == 'M':  # ratio is molar concentration
            mol = ratio * sum(solvent_volume) / 1000  # mol = mol/L * cm3 * 1L/1000cm3
        elif c_type == 'm':  # ratio is molal concentration
            mol = ratio * sum(solvent_weight) /1000  # mol = mol/kg * g * 1kg/1000g
        elif c_type == 'mol':  # ratio is molar
            mol = ratio  # mol = mol
        weight = calculate_molecular_weight(smi) * mol  # g = g/mol * mol
        volume = weight / converting('Density\n(g cm-3)', abb, ri)  # cm3 = g / g/cm3
        salt_mol.append(mol)
        salt_weight.append(weight)
        salt_volume.append(volume)

    # Additive
    additive_parsed = parse_additive(additive, chem_dict)
    additive_weight = []
    additive_volume = []
    additive_mol = []
    for abb, _, ratio, c_type in additive_parsed:
        if c_type == 'wt%':  # ratio is weight percentage
            total_ratio = sum([r for _, _, r, _ in additive_parsed ])/100  # total ratio of additive
            weight = total_ratio / (1-total_ratio) * sum(solvent_weight+salt_weight)  # g 
            volume = weight / converting('Density\n(g cm-3)', abb, ri)  # cm3 = g / g/cm3
            mol = weight / calculate_molecular_weight(smi)  # mol = g / g/mol
            additive_weight.append(weight)
            additive_volume.append(volume)
            additive_mol.append(mol)

    # Total
    total_abbs = [abb for abb, _, _ in solvent_parsed] + [abb for abb, _, _, _ in salt_parsed] + [abb for abb, _, _, _ in additive_parsed]
    total_molecules = [mol for _, mol, _ in solvent_parsed] + [mol for _, mol, _, _ in salt_parsed] + [mol for _, mol, _, _ in additive_parsed]
    total_weight = solvent_weight + salt_weight + additive_weight
    total_volume = solvent_volume + salt_volume + additive_volume
    total_mol = solvent_mol + salt_mol + additive_mol
    # check whether nan exists
    if np.isnan(total_weight).any() or np.isnan(total_mol).any():
    #     # find the index of nan
    #     idx = np.argwhere(np.isnan(total_weight))
    #     abb_nan = np.array(total_abbs)[idx]
    #     raise ValueError(f'NaN exists in total weight, volume, or mol {abb_nan}')
        print('Here is nan' + 88*'*')
        print(total_abbs)
        print(total_weight)
        print(total_volume)
        print(total_mol)

    # Normalize the molar ratio
    total_mol_sum = sum(total_mol)
    total_weight = [weight/total_mol_sum for weight in total_weight]
    total_volume = [vol/total_mol_sum for vol in total_volume]
    total_mol = [mol/total_mol_sum for mol in total_mol]
    # make dictionary which abbs are keys and values are [molecule, weight, volume, mol]
    total_dict = {}
    for abb, molecule, weight, volume, mol in zip(total_abbs, total_molecules, total_weight, total_volume, total_mol):
        total_dict[molecule] = [abb, weight, volume, mol]
    # total_dict['metadata'] = [solvent, salt, additive]
    return total_dict


def y_converter_inverse(y):
    # LCE = -log(1-CE) proposed by PNAS paper
    return 1-10**(-(y))


def y_converter(y):
    # LCE = -log(1-CE) proposed by PNAS paper
    return (-np.log10(1-y))


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TorchMLPRegressor:
    def __init__(self, 
                 hidden_layer_sizes=(100,), 
                 activation='relu', 
                 solver='adam', 
                 alpha=0.0001,
                 learning_rate_init=0.001, 
                 max_iter=200, 
                 batch_size='auto', 
                 tol=1e-4, 
                 verbose=False,
                 random_state=None, 
                 early_stopping=False, 
                 validation_fraction=0.1,
                 n_iter_no_change=10):
        """
        Parameters similar to scikit-learn's MLPRegressor:
          - hidden_layer_sizes: Tuple of hidden layer sizes.
          - activation: Activation function ('relu', 'tanh', 'logistic', 'identity').
          - solver: Optimizer to use ('adam' or 'sgd').
          - alpha: L2 regularization parameter.
          - learning_rate_init: Initial learning rate.
          - max_iter: Maximum number of epochs.
          - batch_size: 'auto' (min(200, n_samples)) or an integer.
          - tol: Tolerance for the optimization.
          - verbose: If True, prints loss per epoch.
          - random_state: Seed for reproducibility.
          - early_stopping: If True, uses a validation split for early stopping.
          - validation_fraction: Fraction of training data for validation if early_stopping is True.
          - n_iter_no_change: Number of epochs with no improvement to wait before stopping.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self._is_fitted = False

    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'logistic':
            return nn.Sigmoid()
        elif self.activation == 'identity':
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _build_model(self, input_dim, output_dim):
        layers = []
        in_dim = input_dim
        act_fn = self._get_activation()
        # Build hidden layers with activation
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn)
            in_dim = hidden_dim
        # Output layer (no activation for regression)
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def fit(self, X, y):
        """
        Trains the MLP on the provided data.
        X should be array-like of shape (n_samples, n_features)
        y should be array-like of shape (n_samples,) or (n_samples, n_outputs)
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples, input_dim = X.shape
        output_dim = y.shape[1]
        
        # Determine batch size
        if self.batch_size == 'auto':
            self.batch_size_ = min(200, n_samples)
        else:
            self.batch_size_ = self.batch_size

        # Set random seed if provided
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Set device (use CUDA if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build the model based on input and output dimensions and move to device
        self._build_model(input_dim, output_dim)
        self.model.to(self.device)
        criterion = nn.MSELoss()
        
        # Select optimizer based on solver parameter; use weight_decay for L2 regularization (alpha)
        if self.solver == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        elif self.solver == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        else:
            raise ValueError(f"Unsupported solver: {self.solver}")

        # Create a validation split if early stopping is enabled
        if self.early_stopping:
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            split = int(n_samples * (1 - self.validation_fraction))
            train_idx, val_idx = indices[:split], indices[split:]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y

        best_val_loss = np.inf
        no_improve_count = 0

        # Training loop
        for epoch in range(self.max_iter):
            # Shuffle training data each epoch
            permutation = np.random.permutation(len(X_train))
            self.model.train()
            epoch_loss = 0.0

            for i in range(0, len(X_train), self.batch_size_):
                indices = permutation[i:i+self.batch_size_]
                # Convert batch data to tensors and move them to the device
                batch_X = torch.from_numpy(X_train[indices]).to(self.device)
                batch_y = torch.from_numpy(y_train[indices]).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(indices)
            epoch_loss /= len(X_train)

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.max_iter}, Training Loss: {epoch_loss:.6f}")

            # Check for early stopping
            if self.early_stopping:
                self.model.eval()
                with torch.no_grad():
                    # Convert validation data to tensors and move to device
                    val_X_tensor = torch.from_numpy(X_val).to(self.device)
                    val_y_tensor = torch.from_numpy(y_val).to(self.device)
                    val_outputs = self.model(val_X_tensor)
                    val_loss = criterion(val_outputs, val_y_tensor).item()
                if self.verbose:
                    print(f"Validation Loss: {val_loss:.6f}")
                # If improvement is observed, reset the counter and save the model state
                if val_loss + self.tol < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                    best_model_state = self.model.state_dict()
                else:
                    no_improve_count += 1
                    if no_improve_count >= self.n_iter_no_change:
                        if self.verbose:
                            print("Early stopping triggered.")
                        # Restore best model and exit training loop
                        self.model.load_state_dict(best_model_state)
                        break
            else:
                # If not using early stopping, you can also stop when training loss is low enough
                if epoch_loss < self.tol:
                    if self.verbose:
                        print("Convergence achieved.")
                    break

        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Predict using the trained model.
        X should be array-like of shape (n_samples, n_features)
        Returns predictions as a NumPy array.
        """
        if not self._is_fitted:
            raise Exception("Model not fitted, call fit() first.")
        X = np.array(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            # Convert input to tensor and move to device
            X_tensor = torch.from_numpy(X).to(self.device)
            predictions = self.model(X_tensor)
            # Move predictions to CPU before converting to numpy
            predictions = predictions.cpu().numpy()
        return predictions
