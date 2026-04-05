from rdkit import Chem
import torch
from torch_geometric.data import Data

def mol_to_graph(smiles_string):
    # 1. Load with RDKit
    mol = Chem.MolFromSmiles(smiles_string)
    
    # 2. Extract Node Features (e.g., Atomic Number)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])
    x = torch.tensor(atom_features, dtype=torch.float)

    # 3. Extract Edge Index (Bonds)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i]) # Undirected graph
    edge_index = torch.tensor(edge_indices).t().contiguous()

    return Data(x=x, edge_index=edge_index)