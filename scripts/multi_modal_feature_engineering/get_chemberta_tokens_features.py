import os
import argparse
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
TASK_DATASETS = {
    'dta': ['davis', 'kiba'],
    'dti': ['davis', 'kiba'],
    'moa': ['activation', 'inhibition'],
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ChembertaTokenModel(nn.Module):
    def __init__(self, model_path="/data01/jsong/chemberta_model"):
        super().__init__()
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, weights_only=False)
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print(f"Hidden size: {self.model.config.hidden_size}")
        
    def extract_token_features(self, smiles_list):
        """
        Core forward pass: no pooling; return last-layer hidden states directly
        """
        with torch.no_grad():
            # Encode
            encodings = self.tokenizer(
                smiles_list,
                truncation=True,
                padding=True,
                max_length=128, # Increase if there are very long SMILES
                return_tensors='pt'
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**encodings)
            
            # last_hidden_state shape: [batch_size, seq_length, hidden_size]
            return outputs.last_hidden_state, encodings['attention_mask']

def get_token_embeddings(smiles_list, model, batch_size=64):
    """
    Batch-extract token-level features and strip padding to save memory
    """
    emb_dict = {}
    model.eval()
    
    # Convert to list to support batch slicing
    smiles_list = list(smiles_list)
    
    # Process in batches for much faster extraction
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Extracting Token Embs"):
        batch_smiles = smiles_list[i:i+batch_size]
        
        with torch.no_grad():
            # last_hidden shape: [batch_size, max_seq_len_in_batch, 384]
            last_hidden, attention_mask = model.extract_token_features(batch_smiles)
            
            for j, smiles in enumerate(batch_smiles):
                # 1. Compute true token length for this SMILES (excluding padding 0s)
                valid_length = attention_mask[j].sum().item()
                
                # 2. Slice the valid sequence.
                # Sliced shape: [valid_length, 384]
                valid_tokens = last_hidden[j, :valid_length, :]
                
                '''
                Note: valid_tokens includes the leading [CLS] and trailing [SEP].
                If the downstream network strictly needs only chemical-structure features, strip them via:
                valid_tokens = last_hidden[j, 1:valid_length-1, :]
                Keeping them provides a "global contextual semantics" anchor for Cross-Attention and is usually preferred.
                '''
                
                # 3. Move to CPU and store in the dictionary
                emb_dict[smiles] = valid_tokens.cpu()
            
    return emb_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Extract ChemBERTa token features')
    parser.add_argument('--task', type=str, required=True, choices=['dta', 'dti', 'moa'])
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

    chemberta_token_dict = get_token_embeddings(compound_iso_smiles, model, batch_size=64)
    save_path = out_dir / f'{task}_chemberta_token_embs.pt'
    torch.save(chemberta_token_dict, save_path)
    print(f'Token-level feature dictionary saved to {save_path}')


if __name__ == "__main__":
    args = parse_args()
    datasets = [args.dataset] if args.dataset else TASK_DATASETS[args.task]
    model = ChembertaTokenModel()
    for ds in datasets:
        run_one(args.task, ds, model)
