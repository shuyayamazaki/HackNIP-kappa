'''
train, val, test split여기서
미리 edge_index 등 그래프 구성도 여기서
jarvis figshare data에서 graph-text pair를 여기서 만들어서 json으로 저장
'''
from jarvis.core.graphs import Graph
from jarvis.core.atoms import Atoms as JAtoms

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse


periodic_table = {'H': 'Hydrogen',
                'He': 'Helium',
                'Li': 'Lithium',
                'Be': 'Beryllium',
                'B': 'Boron',
                'C': 'Carbon',
                'N': 'Nitrogen',
                'O': 'Oxygen',
                'F': 'Fluorine',
                'Ne': 'Neon',
                'Na': 'Sodium',
                'Mg': 'Magnesium',
                'Al': 'Aluminum',
                'Si': 'Silicon',
                'P': 'Phosphorus',
                'S': 'Sulfur',
                'Cl': 'Chlorine',
                'Ar': 'Argon',
                'K': 'Potassium',
                'Ca': 'Calcium',
                'Sc': 'Scandium',
                'Ti': 'Titanium',
                'V': 'Vanadium',
                'Cr': 'Chromium',
                'Mn': 'Manganese',
                'Fe': 'Iron',
                'Co': 'Cobalt',
                'Ni': 'Nickel',
                'Cu': 'Copper',
                'Zn': 'Zinc',
                'Ga': 'Gallium',
                'Ge': 'Germanium',
                'As': 'Arsenic',
                'Se': 'Selenium',
                'Br': 'Bromine',
                'Kr': 'Krypton',
                'Rb': 'Rubidium',
                'Sr': 'Strontium',
                'Y': 'Yttrium',
                'Zr': 'Zirconium',
                'Nb': 'Niobium',
                'Mo': 'Molybdenum',
                'Tc': 'Technetium',
                'Ru': 'Ruthenium',
                'Rh': 'Rhodium',
                'Pd': 'Palladium',
                'Ag': 'Silver',
                'Cd': 'Cadmium',
                'In': 'Indium',
                'Sn': 'Tin',
                'Sb': 'Antimony',
                'Te': 'Tellurium',
                'I': 'Iodine',
                'Xe': 'Xenon',
                'Cs': 'Cesium',
                'Ba': 'Barium',
                'La': 'Lanthanum',
                'Ce': 'Cerium',
                'Pr': 'Praseodymium',
                'Nd': 'Neodymium',
                'Pm': 'Promethium',
                'Sm': 'Samarium',
                'Eu': 'Europium',
                'Gd': 'Gadolinium',
                'Tb': 'Terbium',
                'Dy': 'Dysprosium',
                'Ho': 'Holmium',
                'Er': 'Erbium',
                'Tm': 'Thulium',
                'Yb': 'Ytterbium',
                'Lu': 'Lutetium',
                'Hf': 'Hafnium',
                'Ta': 'Tantalum',
                'W': 'Tungsten',
                'Re': 'Rhenium',
                'Os': 'Osmium',
                'Ir': 'Iridium',
                'Pt': 'Platinum',
                'Au': 'Gold',
                'Hg': 'Mercury',
                'Tl': 'Thallium',
                'Pb': 'Lead',
                'Bi': 'Bismuth',
                'Po': 'Polonium',
                'At': 'Astatine',
                'Rn': 'Radon',
                'Fr': 'Francium',
                'Ra': 'Radium',
                'Ac': 'Actinium',
                'Th': 'Thorium',
                'Pa': 'Protactinium',
                'U': 'Uranium',
                'Np': 'Neptunium',
                'Pu': 'Plutonium',
                'Am': 'Americium',
                'Cm': 'Curium',
                'Bk': 'Berkelium',
                'Cf': 'Californium',
                'Es': 'Einsteinium',
                'Fm': 'Fermium',
                'Md': 'Mendelevium',
                'No': 'Nobelium',
                'Lr': 'Lawrencium',
                'Rf': 'Rutherfordium',
                'Db': 'Dubnium',
                'Sg': 'Seaborgium',
                'Bh': 'Bohrium',
                'Hs': 'Hassium',
                'Mt': 'Meitnerium',
                'Ds': 'Darmstadtium',
                'Rg': 'Roentgenium',
                'Cn': 'Copernicium',
                'Nh': 'Nihonium',
                'Fl': 'Flerovium',
                'Mc': 'Moscovium',
                'Lv': 'Livermorium',
                'Ts': 'Tennessine',
                'Og': 'Oganesson'}


def count_elements(lst):
    '''
    Count elements in a list    
    '''
    element_count = {}
    for elem in lst:
        if elem in element_count:
            element_count[elem] += 1
        else:
            element_count[elem] = 1
    return element_count


def print_element_counts(element_counts):
    '''
    Return a string of elem counts
    '''
    text = 'The unit cell consists of '
    for elem, count in element_counts.items():
        text += f'{count} {periodic_table[elem]}, '
    text = text[:-2] + '.'
    return text


class GenerateGraphData:
    def __init__(self):
        pass
    
    def convert(self, input_df, split):
        '''
        Convert figshare data to json
        '''
        # open figshare data and convert to graph
        # split series using np
        data = np.array_split(input_df, 12)

        results = Parallel(n_jobs=12)(delayed(self.treat_chunk)(chunk) for chunk in tqdm(data))
        results = pd.DataFrame(sum(results, []))
        return results
    
    def treat_chunk(self, chunk):
        result = []
        for _, row in chunk.iterrows():
            item = self.generate_graph(row['cif'])
            result.append(item)
        return result

    def generate_graph(self, cif):
        '''
        Generate graph and text for a single atomic structure
            cif: str
            items: dict
        '''
        jatoms = JAtoms.from_cif(from_string=cif)
        # Generate graph
        dglgraph = Graph.atom_dgl_multigraph(
            jatoms,
            neighbor_strategy="k-nearest",
            cutoff=8.0,
            max_neighbors=12,
            compute_line_graph=False, 
            atom_features='cgcnn'
        )
        item = {}
        edges = dglgraph.edges()
        item['edge_index'] = [edges[0].tolist(), edges[1].tolist()]
        item['num_nodes'] = dglgraph.num_nodes()
        item['node_feat'] = dglgraph.ndata['atom_features'].tolist()
        item['edge_attr'] = dglgraph.edata['r'].tolist()
        
        return item
    

def main(args):
    converter = GenerateGraphData()
    input_df = pd.read_csv(args.data_path)
    converted_df = converter.convert(input_df, args.split)
    output_df = pd.concat([input_df, converted_df], axis=1)  
    output_df.to_parquet(args.output_path)


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='MPContribs_armorphous_diffusivity.csv')
    parser.add_argument('--split', type=int, default=-1)
    parser.add_argument('--output_path', type=str, default='MPContribs_armorphous_diffusivity.parquet')
    args = parser.parse_args()

    print(args)
    main(args)