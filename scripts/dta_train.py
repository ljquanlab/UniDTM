import os
import sys
from pathlib import Path

# Ensure imports and data/ resolve correctly whether launched from project root or scripts/
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
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error
import torch
import pandas as pd
import numpy as np
import random


# Set random seeds for reproducibility; run 5 seeds: 4221, 4555, 3407, 1949, 2026
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

seeds = [4221, 4555, 3407, 1949, 2026]
def train_and_test(dataset, opt, fold=0):
    seed_torch(seed=seeds[fold])
    
    # Load embedding dictionaries
    drug_text_emb_dict_llm = torch.load(f'data/dta/{dataset}/drug_text_embeddings.pt', weights_only=False)
    prot_text_emb_dict_llm = torch.load(f'data/dta/{dataset}/prot_text_embeddings.pt', weights_only=False)
    chemberta_pooling_emb_dict = torch.load(f'data/dta/{dataset}/dta_chemberta_pooling_embs.pt', weights_only=False)
    protein_esm2_pooling_emb_dict = torch.load(f'data/dta/{dataset}/dta_protein_features_pooling.pt', weights_only=False)
    fingerprint_emb_dict = torch.load(f'data/dta/{dataset}/dta_fingerprint_embs.pt', weights_only=False)
    drug_chemberta_emb_dict = torch.load(f'data/dta/{dataset}/dta_chemberta_token_embs.pt', weights_only=False)
    prot_esm2_pad_emb_dict = torch.load(f'data/dta/{dataset}/dta_protein_features_pad.pt', weights_only=False)

    # Load raw train and test data
    train_data = TestbedDataset(root=f"data/dta/{dataset}/{opt}", dataset=f"train")
    test_data = TestbedDataset(root=f"data/dta/{dataset}/{opt}", dataset=f"test")
    
    # Create DataLoaders
    BATCH_SIZE = 256
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
        task='dta',
        dataset_opt=f'{dataset}_{opt}'
    ).to(device)

    # ensemble_model.load_state_dict(torch.load(f'saved_model/dta/result_{fold}/{dataset}/{opt}/best_model.pth', weights_only=False))
    # ensemble_model.load_state_dict(torch.load(f'saved_model/dta/result_{fold}_old/{dataset}/{opt}/best_model.pth', weights_only=False))
    
    # Train on the split train/validation sets
    ensemble_model.fit(train_loader=train_loader, NUM_EPOCHS=200)

    print(f'Testing......')
    # Predict on the test set
    ensemble_model.eval()
    total_true, total_predict = ensemble_model.pred(test_loader=test_loader)

    # Compute metrics
    val_mse = mean_squared_error(total_true, total_predict)
    val_ci = concordance_index(total_true, total_predict)
    pearson_r, _ = pearsonr(total_true, total_predict)
    spearman_rho, _ = spearmanr(total_true, total_predict)

    log_str = f'Test: MSE: {val_mse:.4f}, CI: {val_ci:.4f}, Pearson r: {pearson_r:.4f}, Spearman ρ: {spearman_rho:.4f}'
    print(log_str)

    saved_result_path = f'saved_result/dta/result_{fold}/{dataset}/{opt}'
    os.makedirs(saved_result_path, exist_ok=True)
    # Save predictions
    results_df = pd.DataFrame({
        'true': total_true,
        'pred': total_predict
    })

    results_df.to_csv(f'{saved_result_path}/predictions.csv', index=False)
    print(f"Predictions saved to: {saved_result_path}")


    saved_model_path = f'saved_model/dta/result_{fold}/{dataset}/{opt}'
    os.makedirs(saved_model_path, exist_ok=True)
    # Save model
    torch.save(ensemble_model.state_dict(), f'{saved_model_path}/best_model.pth')
    print(f"Model saved to: {saved_model_path}")


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    datasets = ['davis']
    for fold in range(0, 5): # range(0, 5)
        print(f'======================fold{fold}======================')
        for dataset in datasets:
            for opt in ['cold_drug', 'cold_target', 'cold_drug_target']:
                train_and_test(dataset=dataset, opt=opt, fold=fold)
