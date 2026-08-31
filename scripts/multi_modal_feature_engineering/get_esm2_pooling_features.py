import os
import argparse
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm.auto import tqdm
import pandas as pd
import esm

ROOT = Path(__file__).resolve().parents[2]
TASK_DATASETS = {
    'dta': ['davis', 'kiba'],
    'dti': ['davis', 'kiba'],
    'moa': ['activation', 'inhibition'],
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
esm2_embadings_map = {}


def load_esm2():
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()
    return esm_model, batch_converter


def prot_embedding(sequence, esm_model, batch_converter):
    global esm2_embadings_map
    if esm2_embadings_map.get(sequence) is not None:
        return esm2_embadings_map[sequence]

    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33])  # 33 is the last layer of ESM2-650M
        token_reps = results["representations"][33]
        # Take the CLS token embedding as the whole-sequence vector
        cls_emb = token_reps[:, 0, :]

    esm2_embadings_map[sequence] = cls_emb
    return cls_emb


def create_pooling_pt_file(unique_sequences, save_path, esm_model, batch_converter):
    esm2_features = {}

    with torch.no_grad():
        for seq in tqdm(unique_sequences, desc="Extracting ESM2 pooling features"):
            pooling_emb = prot_embedding(seq, esm_model, batch_converter)
            esm2_features[seq] = pooling_emb

    torch.save(esm2_features, save_path)
    print(f"Save complete! Features saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Extract ESM2 pooling features')
    parser.add_argument('--task', type=str, required=True, choices=['dta', 'dti', 'moa'])
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name. If omitted, process all datasets for the task.')
    return parser.parse_args()


def run_one(task, dataset, esm_model, batch_converter):
    global esm2_embadings_map
    esm2_embadings_map = {}

    cold_dir = ROOT / 'data' / task / dataset / 'cold_drug'
    out_dir = ROOT / 'data' / task / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = cold_dir / 'train.csv'
    test_csv = cold_dir / 'test.csv'
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f'Missing {train_csv} or {test_csv}')

    protein_seq = []
    protein_seq += list(pd.read_csv(train_csv)['Target'])
    protein_seq += list(pd.read_csv(test_csv)['Target'])
    protein_seq = set(protein_seq)
    print(f'[{task}/{dataset}] Total unique proteins: {len(protein_seq)}')

    save_path = out_dir / f'{task}_protein_features_pooling.pt'
    create_pooling_pt_file(list(protein_seq), save_path, esm_model, batch_converter)


if __name__ == "__main__":
    args = parse_args()
    datasets = [args.dataset] if args.dataset else TASK_DATASETS[args.task]
    print('Using device:', device)
    esm_model, batch_converter = load_esm2()
    for ds in datasets:
        run_one(args.task, ds, esm_model, batch_converter)
