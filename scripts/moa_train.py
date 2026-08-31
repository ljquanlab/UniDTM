'''Train only the three activation experimental settings'''
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_FEAT_DIR = _SCRIPT_DIR / 'multi_modal_feature_engineering'
if str(_FEAT_DIR) not in sys.path:
    sys.path.insert(0, str(_FEAT_DIR))
os.chdir(_ROOT_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from model import EnsembleModel
from get_drug_graph_esm2_redidue import TestbedDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import pandas as pd
import random
import numpy as np


def seed_torch(seed=4221):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seeds = [4221, 4555, 3407, 1949, 4321]
def train_and_test(dataset, opt, fold=0):
    # Set random seed for reproducibility
    seed_torch(seed=seeds[fold])
    
    # Load embedding dictionaries
    drug_text_emb_dict_llm = torch.load(f'data/moa/{dataset}/drug_text_embeddings.pt', weights_only=False)
    prot_text_emb_dict_llm = torch.load(f'data/moa/{dataset}/prot_text_embeddings_qwen.pt', weights_only=False)
    chemberta_pooling_emb_dict = torch.load(f'data/moa/{dataset}/moa_chemberta_pooling_embs.pt', weights_only=False)
    protein_esm2_pooling_emb_dict = torch.load(f'data/moa/{dataset}/moa_protein_features_pooling.pt', weights_only=False)
    fingerprint_emb_dict = torch.load(f'data/moa/{dataset}/moa_fingerprint_embs.pt', weights_only=False)
    drug_chemberta_emb_dict = torch.load(f'data/moa/{dataset}/moa_chemberta_token_embs.pt', weights_only=False)
    prot_esm2_pad_emb_dict = torch.load(f'data/moa/{dataset}/moa_protein_features_pad.pt', weights_only=False)

    # Load raw train and test data
    train_data = TestbedDataset(root=f"data/moa/{dataset}/{opt}", dataset=f"train")
    test_data = TestbedDataset(root=f"data/moa/{dataset}/{opt}", dataset=f"test")

    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Create DataLoaders
    BATCH_SIZE = 128
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Build ensemble model
    ensemble_model = EnsembleModel(
        esm2_pad_dict=prot_esm2_pad_emb_dict,
        esm2_pooling_dict=protein_esm2_pooling_emb_dict,
        drug_text_emb_dict=drug_text_emb_dict_llm,
        prot_text_emb_dict=prot_text_emb_dict_llm,
        chemberta_pad_emb_dict=drug_chemberta_emb_dict,
        chemberta_pooling_emb_dict=chemberta_pooling_emb_dict,
        fingerprint_emb_dict=fingerprint_emb_dict,
        device=device,
        task='moa',
        dataset_opt=f'{dataset}_{opt}'
    ).to(device)

    # Train on the split train/validation sets
    ensemble_model.fit(train_loader=train_loader, NUM_EPOCHS=200)
    # ensemble_model.load_state_dict(torch.load(f'saved_model/moa/fold_{fold}/{dataset}/{opt}/best_model.pth', weights_only=False))

    print(f'Testing......')
    # Predict on the test set
    total_true, total_probs = ensemble_model.pred(test_loader=test_loader)

    test_auc = roc_auc_score(total_true, total_probs)
    test_aupr = average_precision_score(total_true, total_probs)

    log_str = f'Test -> AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}'
    print(log_str)

    # Save predictions
    results_df = pd.DataFrame({
        'true': total_true,
        'pred': total_probs
    })
    result_save_path = f'saved_result/moa/fold_{fold}/{dataset}/{opt}'
    os.makedirs(result_save_path, exist_ok=True)
    save_path = f'predictions.csv'
    results_df.to_csv(os.path.join(result_save_path, save_path), index=False)
    print(f"Predictions saved to: {save_path}")

    # Save model
    model_saved_path = f'saved_model/moa/fold_{fold}/{dataset}/{opt}'
    os.makedirs(model_saved_path, exist_ok=True)
    torch.save(ensemble_model.state_dict(), f'{model_saved_path}/best_model.pth')
    print(f"Model saved to: best_model.pth")


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    datasets = ['activation']
    for fold in range(4, 5):
        print(f'======================fold{fold}======================')
        for dataset in datasets:
            for opt in ['cold_drug_target']: # 'cold_drug', 'cold_target', 
                train_and_test(dataset=dataset, opt=opt, fold=fold)
