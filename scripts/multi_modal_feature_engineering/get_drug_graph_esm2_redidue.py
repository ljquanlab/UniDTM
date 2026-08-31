import os
import argparse
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from tqdm.auto import tqdm
import pandas as pd
import esm

RDLogger.DisableLog('rdApp.warning')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ROOT = Path(__file__).resolve().parents[2]
TASK_DATASETS = {
    'dta': ['davis', 'kiba'],
    'dti': ['davis', 'kiba'],
    'moa': ['activation', 'inhibition'],
}
SPLITS = ['cold_drug', 'cold_target', 'cold_drug_target']


# Get node features (nodes are atoms)
def atom_features(atom):
    # 1. One-hot encoding of the atom symbol
    def one_hot_encoding_unk(x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]
    atom_symbol = one_hot_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si',
                           'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
                            'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li',
                            'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])

    # 2. Atom degree, also one-hot encoded
    def one_hot_encoding(x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]
    atom_degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # 3. Total number of hydrogens attached to the atom
    atom_hs = one_hot_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # 4. Implicit valence [how many more bonds the atom can form under standard valence rules]
    atom_imp_val = one_hot_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # 5. Formal charge [valence electrons - (lone pair electrons + 1/2 * bonding electrons)]
    atom_fo_ch = one_hot_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])

    # 6. Hybridization state: SP (linear), SP2 (trigonal planar), SP3 (tetrahedral), etc.
    '''
    Common HybridizationType enum values:
    rdkit.Chem.rdchem.HybridizationType.SP
    rdkit.Chem.rdchem.HybridizationType.SP2
    rdkit.Chem.rdchem.HybridizationType.SP3
    rdkit.Chem.rdchem.HybridizationType.SP3D
    rdkit.Chem.rdchem.HybridizationType.SP3D2
    rdkit.Chem.rdchem.HybridizationType.S
    rdkit.Chem.rdchem.HybridizationType.Unspecified
    '''
    atom_hybri = one_hot_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                             Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                             Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2])

    # 7. Whether the atom is part of an aromatic system
    atom_aroma = [1 if atom.GetIsAromatic() else 0]

    # 8. Whether the atom is in a ring
    atom_in_ring = [1 if atom.IsInRing() else 0]

    # ========== Fused advanced features ==========
    # 7. Possible chiral center flag
    chirality_possible = [1 if atom.HasProp('_ChiralityPossible') else 0]
    # 8. R/S absolute chirality (critical for drug-target affinity)
    try:
        cip_code = atom.GetProp('_CIPCode')
        cip_encoded = one_hot_encoding_unk(cip_code, ['R', 'S', 'Unknown'])
    except KeyError:
        cip_encoded = [0, 0, 1]  # Fill Unknown when chirality info is missing
    # 9. Atomic mass (scaled by 0.01 to avoid numerical explosion)
    mass = [atom.GetMass() * 0.01]

    # 9. Return concatenation of all attributes
    '''
    atom symbol size: 44
    atom degree feature size 11
    total H feature size 11
    implicit valence feature size 11
    formal charge feature size 5
    hybridization feature size 5
    atom_aroma size 1
    atom_in_ring size 1
        :return size(89)
    '''
    return np.array(atom_symbol + atom_degree + atom_hs + atom_imp_val + atom_fo_ch + atom_hybri +
                    atom_aroma + atom_in_ring + chirality_possible + cip_encoded + mass, dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [0, 0, 0, 0, bond.GetBondTypeAsDouble()]
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_feats = [1, 0, 0, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_feats = [0, 1, 0, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_feats = [0, 0, 1, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.AROMATIC:
        bond_feats = [0, 0, 0, 1, bond.GetBondTypeAsDouble()]

    # ========== Fused advanced features ==========
    # 1. Whether the bond is conjugated (electron delocalization network)
    is_conjugated = [1 if bond.GetIsConjugated() else 0]
    # 2. Whether the bond is in a ring (affects local molecular rigidity)
    is_in_ring = [1 if bond.IsInRing() else 0]
    # 3. Cis/trans stereochemistry (E/Z)
    stereo = str(bond.GetStereo())
    stereo_encoded = [1 if stereo == "STEREONONE" else 0,
                      1 if stereo == "STEREOANY" else 0,
                      1 if stereo == "STEREOZ" else 0,
                      1 if stereo == "STEREOE" else 0]
    return np.array(bond_feats + is_conjugated + is_in_ring + stereo_encoded, dtype=np.float32)


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:  # Validity check to avoid crashes
        return 0, [], [], []

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        features.append(atom_features(atom))

    # ========== Skip NetworkX; build a bidirectional graph with native lists ==========
    edge_index = []
    edge_feats = []

    for bond in mol.GetBonds():
        b_feat = bond_features(bond)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        # GNNs typically need an undirected equivalent: add both forward and reverse edges
        edge_index.append([u, v])
        edge_feats.append(b_feat)

        edge_index.append([v, u])
        edge_feats.append(b_feat)

    return c_size, features, edge_index, edge_feats


# ================= 1. Optimized ESM feature extraction =================
# Max truncate length to avoid ESM OOM on very long sequences (ESM2 default max ~1024)
MAX_SEQ_LEN = 1000


def get_unique_prot_embeddings(unique_sequences):
    """
    Extract ESM features for deduplicated proteins in batch; return a dictionary
    """
    # Load ESM2 only when this function is called (not at import time)
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)
    esm_model.eval()  # Enter inference mode
    batch_converter = alphabet.get_batch_converter()

    esm2_emb_map = {}

    # Show progress with tqdm
    for seq in tqdm(unique_sequences, desc="Extracting ESM2 Embeddings"):
        # Truncate overly long sequences
        trunc_seq = seq[:MAX_SEQ_LEN]

        data = [("protein", trunc_seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33])
            token_reps = results["representations"][33]

            # Extract true amino-acid residue features [SeqLen, 1280]
            # Store in a CPU memory dict to save GPU VRAM
            residue_emb = token_reps[0, 1: len(trunc_seq) + 1, :].cpu()

        esm2_emb_map[seq] = residue_emb

    return esm2_emb_map


# ================= 2. Optimized Dataset builder =================
class TestbedDataset(InMemoryDataset):
    # Removed the xt argument; we no longer stuff the huge ESM Tensor into graph data
    def __init__(self, root='data', dataset='davis',
                 xd=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, ps_sequence=None):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], map_location=device, weights_only=False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process_data(xd, y, smile_graph, ps_sequence)
            self.data, self.slices = torch.load(self.processed_paths[0], map_location=device, weights_only=False)

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def process_data(self, xd, y, smile_graph, ps_sequence):
        data_list = []
        data_len = len(xd)
        for i in tqdm(range(data_len), desc=f'Building PyG Dataset {self.dataset}'):
            smiles = xd[i]
            protein_sequence = ps_sequence[i]
            labels = y[i]

            c_size, features, edge_index, edge_feats = smile_graph[smiles]

            # Skip invalid SMILES
            if c_size == 0:
                continue

            # Process molecule atoms
            EDGE_FEAT_DIM = 11  # Feature dim returned by bond_features
            if len(edge_index) == 0:
                # 1. Undirected graph with 0 edges
                edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
                # 2. 0 edges, each with 11 features
                edge_attr_tensor = torch.empty((0, EDGE_FEAT_DIM), dtype=torch.float)
            else:
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr_tensor = torch.tensor(edge_feats, dtype=torch.float)

            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=edge_index_tensor,
                edge_attr=edge_attr_tensor,
                y=torch.FloatTensor([labels])
            )

            # Store only the sequence text string — extremely lightweight!
            GCNData.smiles = smiles
            GCNData.protein_sequence = protein_sequence
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            data_list.append(GCNData)

        print('Data preparation Done!. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build molecular graphs + residue ESM2 + PyG processed/{train,test}.pt for all splits'
    )
    parser.add_argument('--task', type=str, required=True, choices=['dta', 'dti', 'moa'])
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name. If omitted, process all datasets for the task.')
    return parser.parse_args()


def run_one(task, dataset):
    """
    Use cold_drug to collect all drugs/proteins and extract shared residue ESM2.
    Then build processed train/test.pt for each experimental split.
    """
    data_root = ROOT / 'data' / task / dataset
    cold_dir = data_root / 'cold_drug'
    train_csv = cold_dir / 'train.csv'
    test_csv = cold_dir / 'test.csv'
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f'Missing {train_csv} or {test_csv}')

    df_cold_train = pd.read_csv(train_csv)
    df_cold_test = pd.read_csv(test_csv)

    # 1. Collect all unique drugs and proteins from cold_drug (covers all entities)
    all_drugs = set(df_cold_train['Drug']).union(set(df_cold_test['Drug']))
    all_prots = set(df_cold_train['Target']).union(set(df_cold_test['Target']))
    print(f'[{task}/{dataset}] unique drugs={len(all_drugs)}, proteins={len(all_prots)}')

    # 2. Build molecular graph dictionary once
    smile_graph = {}
    for smile in tqdm(all_drugs, desc=f'[{dataset}] Building Molecular Graphs'):
        smile_graph[smile] = smile_to_graph(smile)

    # 3. Residue-level ESM2 dictionary -> data/{task}/{dataset}/{task}_protein_features_pad.pt
    data_root.mkdir(parents=True, exist_ok=True)
    dict_path = data_root / f'{task}_protein_features_pad.pt'
    if not dict_path.exists():
        print(f'Extracting ESM2 residue dictionary for {len(all_prots)} proteins...')
        esm2_dict = get_unique_prot_embeddings(all_prots)
        torch.save(esm2_dict, dict_path)
        print(f'ESM2 residue dictionary saved to {dict_path}')
    else:
        print(f'Found existing ESM2 residue dictionary: {dict_path}')

    # 4. Build PyG Dataset for each experimental setting
    for split in SPLITS:
        split_dir = data_root / split
        split_train = split_dir / 'train.csv'
        split_test = split_dir / 'test.csv'
        if not split_train.exists() or not split_test.exists():
            print(f'Skip missing split: {split_dir}')
            continue

        df_train = pd.read_csv(split_train)
        df_test = pd.read_csv(split_test)
        (split_dir / 'processed').mkdir(parents=True, exist_ok=True)

        print(f'[{task}/{dataset}/{split}] Preparing train.pt ...')
        TestbedDataset(
            root=str(split_dir), dataset='train',
            xd=list(df_train['Drug']), y=list(df_train['Y']),
            smile_graph=smile_graph, ps_sequence=list(df_train['Target'])
        )

        print(f'[{task}/{dataset}/{split}] Preparing test.pt ...')
        TestbedDataset(
            root=str(split_dir), dataset='test',
            xd=list(df_test['Drug']), y=list(df_test['Y']),
            smile_graph=smile_graph, ps_sequence=list(df_test['Target'])
        )


# ================= 3. Main pipeline =================
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    print('Using device:', device)

    args = parse_args()
    datasets = [args.dataset] if args.dataset else TASK_DATASETS[args.task]
    for ds in datasets:
        run_one(args.task, ds)
