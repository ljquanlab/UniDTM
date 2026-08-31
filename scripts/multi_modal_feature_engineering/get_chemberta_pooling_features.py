import os
import argparse
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
TASK_DATASETS = {
    'dta': ['davis', 'kiba'],
    'dti': ['davis', 'kiba'],
    'moa': ['activation', 'inhibition'],
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ChembertaModel(nn.Module):
    def __init__(self, model_path="/data01/jsong/chemberta_model", dropout=0.1, output_dim=256):
        super().__init__()
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, weights_only=False)
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print(f"Hidden size: {self.model.config.hidden_size}")
        
    def extract_features(self, smiles_list, pooling_strategy="mean"):
        """
        Extract features from SMILES strings
        
        Args:
            smiles_list: list of SMILES strings
            batch_size: batch size
            pooling_strategy: pooling strategy ["cls", "mean", "max"]
            
        Returns:
            features: feature matrix (n_samples, hidden_size)  # vector, 384-d
        """
        all_features = []
        
        with torch.no_grad():
            # Encode
            encodings = self.tokenizer(
                smiles_list,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**encodings)
            last_hidden_state = outputs.last_hidden_state

            attention_mask = encodings['attention_mask']
            # --- Prepare expanded mask ---
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            if pooling_strategy == "hybrid":
                # 1. Mean pooling
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                mean_feat = sum_embeddings / sum_mask
                
                # 2. Max pooling
                # Set padding positions to a large negative so they do not affect max
                masked_hidden = last_hidden_state.clone()
                masked_hidden[mask_expanded == 0] = -1e9
                max_feat, _ = torch.max(masked_hidden, 1)
                
                # 3. Concatenate
                # Result dim goes from 768 to 1536 (768 * 2)
                batch_features = torch.cat([mean_feat, max_feat], dim=-1)
            
            # Pooling strategies
            elif pooling_strategy == "cls":
                # Use [CLS] token features
                batch_features = last_hidden_state[:, 0, :]
            elif pooling_strategy == "mean":
                # Mean pooling
                attention_mask = encodings['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_features = sum_embeddings / sum_mask
            elif pooling_strategy == "max":
                # Max pooling
                attention_mask = encodings['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                last_hidden_state[input_mask_expanded == 0] = -1e9
                batch_features, _ = torch.max(last_hidden_state, 1)
            
            all_features.append(batch_features.cpu())
        
        # Concatenate features from all batches
        features = torch.cat(all_features, dim=0)
        return features

def randomize_smiles(smiles):
    """Generate alternative SMILES for the same molecule to improve generalization"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return smiles
    # Randomize atom order and generate a new SMILES
    return Chem.MolToSmiles(mol, doRandom=True, canonical=False)

def get_robust_embeddings(smiles_list, model, num_variants=5):
    """
    For each SMILES, generate multiple variants and average their features
    """
    emb_dict = {}
    model.eval()
    
    for s in tqdm(smiles_list, desc="Extracting Robust Embs"):
        # 1. Prepare variant list (including the original SMILES)
        variants = {s}  # Use a set for deduplication
        for i in range(0, 50):
            if len(variants) < num_variants:
                variants.add(randomize_smiles(s))
            else:
                break
        
        # 2. Extract features for this variant set (via extract_features)
        with torch.no_grad():
            variant_features = model.extract_features(list(variants), pooling_strategy="cls")

        # 3. Average the 5 vectors to obtain a "holographic" feature for the drug
        avg_feature = torch.mean(variant_features, dim=0)
        emb_dict[s] = avg_feature.cpu()
        
    return emb_dict

def get_embeddings(smiles_list, model):
    """
    For each SMILES, generate multiple variants and average their features
    """
    emb_dict = {}
    model.eval()
    
    for s in tqdm(smiles_list, desc="Extracting Robust Embs"):
        variants = [s]
        with torch.no_grad():
            variant_features = model.extract_features(variants, pooling_strategy="cls")
        emb_dict[s] = torch.mean(variant_features, dim=0)
        
    return emb_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Extract ChemBERTa pooling features')
    parser.add_argument('--task', type=str, required=True, choices=['dta', 'dti', 'moa'],
                        help='Task name: dta / dti / moa')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name. If omitted, process all datasets for the task.')
    return parser.parse_args()


def run_one(task, dataset, model):
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

    save_path = out_dir / f'{task}_chemberta_pooling_embs.pt'
    chemberta_dict = get_robust_embeddings(list(compound_iso_smiles), model)
    torch.save(chemberta_dict, save_path)
    print(f'Pooling feature dictionary saved to {save_path}')


if __name__ == "__main__":
    args = parse_args()
    datasets = [args.dataset] if args.dataset else TASK_DATASETS[args.task]
    model = ChembertaModel()
    for ds in datasets:
        run_one(args.task, ds, model)
