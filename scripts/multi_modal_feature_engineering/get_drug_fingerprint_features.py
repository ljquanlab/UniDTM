import argparse
from pathlib import Path

import torch
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
import pandas as pd

RDLogger.DisableLog('rdApp.warning')

ROOT = Path(__file__).resolve().parents[2]
TASK_DATASETS = {
    'dta': ['davis', 'kiba'],
    'dti': ['davis', 'kiba'],
    'moa': ['activation', 'inhibition'],
}


def save_fingerprints(smiles_list, save_path):
    """
    Convert a SMILES list to Morgan fingerprints and save as a dictionary
    """
    fp_dict = {}
    for smiles in tqdm(smiles_list, desc="Extracting molecular fingerprints"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # radius=2 is equivalent to ECFP4
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_tensor = torch.from_numpy(np.array(fp))
            fp_dict[smiles] = fp_tensor
        else:
            # Zero-pad for invalid SMILES
            fp_dict[smiles] = torch.zeros(2048)

    torch.save(fp_dict, save_path)
    print(f"Fingerprint extraction done, saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Extract Morgan fingerprint features')
    parser.add_argument('--task', type=str, required=True, choices=['dta', 'dti', 'moa'])
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name. If omitted, process all datasets for the task.')
    return parser.parse_args()


def run_one(task, dataset):
    cold_dir = ROOT / 'data' / task / dataset / 'cold_drug'
    out_dir = ROOT / 'data' / task / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = cold_dir / 'train.csv'
    test_csv = cold_dir / 'test.csv'
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f'Missing {train_csv} or {test_csv}')

    compound_iso_smiles = []
    compound_iso_smiles += list(pd.read_csv(train_csv)['Drug'])
    compound_iso_smiles += list(pd.read_csv(test_csv)['Drug'])
    compound_iso_smiles = set(compound_iso_smiles)
    print(f'[{task}/{dataset}] Total unique SMILES: {len(compound_iso_smiles)}')

    save_path = out_dir / f'{task}_fingerprint_embs.pt'
    save_fingerprints(list(compound_iso_smiles), save_path)


if __name__ == "__main__":
    args = parse_args()
    datasets = [args.dataset] if args.dataset else TASK_DATASETS[args.task]
    for ds in datasets:
        run_one(args.task, ds)
