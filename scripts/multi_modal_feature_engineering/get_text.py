import time
import argparse
from pathlib import Path

from openai import OpenAI
import os
import pandas as pd
from tqdm.auto import tqdm
import requests

ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = ROOT / 'prompts'
TASK_DATASETS = {
    'dta': ['davis', 'kiba'],
    'dti': ['davis', 'kiba'],
    'moa': ['activation', 'inhibition'],
}

base_url = "https://integrate.api.nvidia.com/v1"
api_key = ""

def get_drug_text_emb(smiles):
    with open(_PROMPTS_DIR / 'prompt_sys.txt', 'r', encoding='utf-8') as file:
        prompt_sys = file.read()
    with open(_PROMPTS_DIR / 'prompt_user.txt', 'r', encoding='utf-8') as file:
        prompt_user = file.read()

    client_text = OpenAI(
        base_url = base_url,
        api_key = api_key
    )

    completion = client_text.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"content": prompt_sys,"role":"system"},
            {"content":f"SMILES: {smiles}\n{prompt_user}","role":"user"}
        ],
        temperature=1
    )

    desc_text = completion.choices[0].message.content

    return desc_text


def get_uniprot_function_text(uniprot_id):
    if uniprot_id is None or '' == uniprot_id:
        return "No function description available."
    """
    Fetch plain-text function description for a target protein via UniProt ID
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    try:
        response = requests.get(url, timeout=10)
        # Status 200 means the protein was found
        if response.status_code == 200:
            data = response.json()
            
            # Iterate comments looking for "FUNCTION" type descriptions
            if 'comments' in data:
                for comment in data['comments']:
                    if comment.get('commentType') == 'FUNCTION':
                        # Extract the text content
                        texts = [text_obj['value'] for text_obj in comment.get('texts', [])]
                        return " ".join(texts)
            
            return "No function description available."
        else:
            return "No function description available."
            
    except Exception as e:
        return "No function description available."

def get_protein_text_emb(uniprot_id, sequence):
    desc_text = get_uniprot_function_text(uniprot_id)
    if "No function description available." == desc_text:
        with open(_PROMPTS_DIR / 'prompt_sys_protein.txt', 'r', encoding='utf-8') as file:
            prompt_sys = file.read()
        with open(_PROMPTS_DIR / 'prompt_user_protein.txt', 'r', encoding='utf-8') as file:
            prompt_user = file.read()

        client_text = OpenAI(
            base_url = base_url,
            api_key = api_key
        )

        completion = client_text.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"content": prompt_sys,"role":"system"},
                {"content":f'''{prompt_user}
                - UniProt ID: {uniprot_id}
                - Amino Acid Sequence: {sequence}
                Generated Description:''',"role":"user"}
            ],
            temperature=1
        )

        desc_text = completion.choices[0].message.content

    return desc_text


def parse_args():
    parser = argparse.ArgumentParser(description='Generate drug/protein text descriptions via LLM')
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

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # 1. Collect all SMILES and deduplicate
    drug_list, protein_id_list = [], []
    drug_list.extend(df['Drug'].tolist())
    protein_id_list.extend([(row.Target_ID, row.Target) for row in df.itertuples()])

    drug_set = set(drug_list)
    protein_set = set(protein_id_list)
    print(f'[{task}/{dataset}] Total unique SMILES: {len(drug_set)}, unique proteins: {len(protein_set)}')

    # 2. Drug descriptions
    output_file = out_dir / 'drug_descriptions.csv'
    existing_smiles = set()
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file)
            existing_smiles = set(existing_df['SMILES'].tolist())
            print(f"Found {len(existing_smiles)} already processed SMILES")
        except Exception as e:
            print(f"Error reading existing file: {e}")

    smiles_to_process = [s for s in drug_set if s not in existing_smiles]
    print(f"Need to process {len(smiles_to_process)} new SMILES")
    if len(smiles_to_process) == 0:
        print("All SMILES have already been processed!")
    else:
        file_exists = output_file.exists()
        mode = 'a' if file_exists else 'w'
        drug_not_found = []
        for smiles in tqdm(smiles_to_process, desc='Drug descriptions'):
            answer_content = get_drug_text_emb(smiles)
            if ('Target not found in UniProt.' in answer_content or 'No function description available' in answer_content):
                drug_not_found.append(smiles)
                continue
            time.sleep(0.3)

            new_row = pd.DataFrame([{
                "SMILES": smiles,
                "Description": answer_content
            }])
            new_row.to_csv(output_file, mode=mode, index=False, header=not file_exists)
            if mode == 'w':
                mode = 'a'
                file_exists = True
        print(f"Done drugs! Processed {len(smiles_to_process)}, unmatched {len(drug_not_found)}.")
        for nf in drug_not_found:
            print(nf)

    # 3. Protein descriptions
    output_file = out_dir / 'protein_descriptions.csv'
    existing_proteins = set()
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file)
            existing_proteins = set(existing_df['Target'].tolist())
            print(f"Found {len(existing_proteins)} already processed proteins")
        except Exception as e:
            print(f"Error reading existing file: {e}")

    protein_ids_to_process = [(i, s) for i, s in protein_set if s not in existing_proteins]
    print(f"Need to process {len(protein_ids_to_process)} new proteins")
    if len(protein_ids_to_process) == 0:
        print("All proteins have already been processed!")
        return

    file_exists = output_file.exists()
    mode = 'a' if file_exists else 'w'
    not_found = []
    print("Starting protein description search...")
    for target_id, target in tqdm(protein_ids_to_process, desc='Protein descriptions'):
        text = get_protein_text_emb(target_id, target)
        if ('Target not found in UniProt.' in text or 'No function description available' in text):
            not_found.append(target_id)
            continue
        time.sleep(0.3)

        new_row = pd.DataFrame([{
            "Target": target,
            "Description": text
        }])
        new_row.to_csv(output_file, mode=mode, index=False, header=not file_exists)
        if mode == 'w':
            mode = 'a'
            file_exists = True

    print(f"Done proteins! Processed {len(protein_ids_to_process)}, unmatched {len(not_found)}.")
    for nf in not_found:
        print(nf)


if __name__ == "__main__":
    args = parse_args()
    datasets = [args.dataset] if args.dataset else TASK_DATASETS[args.task]
    for ds in datasets:
        run_one(args.task, ds)
