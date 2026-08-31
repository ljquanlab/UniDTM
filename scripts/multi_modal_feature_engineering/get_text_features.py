import argparse
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm
import torch

ROOT = Path(__file__).resolve().parents[2]
TASK_DATASETS = {
    'dta': ['davis', 'kiba'],
    'dti': ['davis', 'kiba'],
    'moa': ['activation', 'inhibition'],
}

client = OpenAI(
  api_key="",
  base_url="https://integrate.api.nvidia.com/v1"
)


def embed_text(desc_text):
    response = client.embeddings.create(
        input=[desc_text],
        model="nvidia/llama-nemotron-embed-1b-v2",
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "NONE", "dimensions": 1024}
    )
    return response.data[0].embedding


def parse_args():
    parser = argparse.ArgumentParser(description='Embed drug/protein text descriptions')
    parser.add_argument('--task', type=str, required=True, choices=['dta', 'dti', 'moa'])
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name. If omitted, process all datasets for the task.')
    return parser.parse_args()


def run_one(task, dataset):
    out_dir = ROOT / 'data' / task / dataset
    drug_csv = out_dir / 'drug_descriptions.csv'
    prot_csv = out_dir / 'protein_descriptions.csv'
    if not drug_csv.exists() or not prot_csv.exists():
        raise FileNotFoundError(
            f'Missing description CSVs under {out_dir}. Run get_text.py --task {task} --dataset {dataset} first.'
        )

    df_drug = pd.read_csv(drug_csv)
    df_prot = pd.read_csv(prot_csv)
    print(f'[{task}/{dataset}] drug: {len(df_drug)}, protein: {len(df_prot)}')

    prot_text_embeddings = {}
    for index, row in tqdm(df_prot.iterrows(), total=len(df_prot), desc='Protein Embedding'):
        seq = str(row['Target'])
        desc_text = row['Description']
        prot_text_embeddings[seq] = embed_text(desc_text)

    # MoA train script expects prot_text_embeddings_qwen.pt
    prot_out = out_dir / ('prot_text_embeddings_qwen.pt' if task == 'moa' else 'prot_text_embeddings.pt')
    torch.save(prot_text_embeddings, prot_out)
    print(f"Saved embeddings for {len(prot_text_embeddings)} proteins -> {prot_out}")

    drug_text_embeddings = {}
    for index, row in tqdm(df_drug.iterrows(), total=len(df_drug), desc='Drug Embedding'):
        smiles = str(row['SMILES'])
        desc_text = row['Description']
        drug_text_embeddings[smiles] = embed_text(desc_text)

    drug_out = out_dir / 'drug_text_embeddings.pt'
    torch.save(drug_text_embeddings, drug_out)
    print(f"Saved embeddings for {len(drug_text_embeddings)} drugs -> {drug_out}")


if __name__ == "__main__":
    args = parse_args()
    datasets = [args.dataset] if args.dataset else TASK_DATASETS[args.task]
    for ds in datasets:
        run_one(args.task, ds)
