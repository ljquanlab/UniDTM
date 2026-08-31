from model_graphmol import ModelGraphMol
from model_llm import ModelLLM
from model_pooling import ModelPooling
from model_premol import ModelPreMol
import torch.nn as nn
import torch
from torch.optim import Adam
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import copy
import os


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0):  # alpha upweights the positive class (1)
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits are the raw (pre-sigmoid) network outputs
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # predicted confidence

        # Positive samples weighted by alpha, negatives by 1-alpha
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class CauchyLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        """
        gamma: scale parameter.
               Larger gamma makes the model more tolerant of large residuals (approaches MSE);
               smaller gamma makes it more sensitive to outliers and more inclined to downweight large residuals.
               For DTA (labels typically in 5~15), gamma in 0.5 to 2.0 is recommended.
        """
        super(CauchyLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        residual = pred - target
        loss = torch.log(1.0 + (residual / self.gamma) ** 2)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class EarlyStopping:
    """Early stopping: halt training when the validation metric stops improving"""

    def __init__(self, patience=5, mode='min', delta=1e-5, verbose=True):
        """
        Args:
            patience: number of epochs allowed without improvement
            mode: 'min' means lower is better; 'max' means higher is better
            delta: minimum change to count as improvement
            verbose: whether to print messages
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """Call after each validation; returns (should_stop, best_was_updated)"""
        best_is_update = False
        if self.best_score is None:
            best_is_update = True
            self.best_score = score
        else:
            if self._is_improvement(score):
                best_is_update = True
                self.best_score = score
                self.counter = 0
                if self.verbose:
                    print(f'Validation metric improved to {score:.4f}.')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'No improvement. Counter: {self.counter}/{self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop, best_is_update

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.delta
        else:
            return score > self.best_score + self.delta


class EnsembleModel(nn.Module):
    def __init__(self, esm2_pad_dict, esm2_pooling_dict, drug_text_emb_dict, prot_text_emb_dict, chemberta_pad_emb_dict,
                 chemberta_pooling_emb_dict, fingerprint_emb_dict, device, task, dataset_opt):
        super().__init__()
        self.device = device
        assert task in ['dta', 'dti', 'moa'], 'task must be contained in [dta, dti, moa]'
        self.task = task
        self.dataset_opt = dataset_opt

        self.model_graphmol = ModelGraphMol(esm2_dict=esm2_pad_dict, device=device).to(device)
        self.model_premol = ModelPreMol(
            chemberta_dict=chemberta_pad_emb_dict, esm2_dict=esm2_pad_dict, device=device
        ).to(device)
        self.model_llm = ModelLLM(
            llm_emb_dict=drug_text_emb_dict, prot_text_emb_dict=prot_text_emb_dict, device=device
        ).to(device)
        self.model_pooling = ModelPooling(
            chemberta_emb_dict=chemberta_pooling_emb_dict,
            llm_emb_dict=drug_text_emb_dict,
            fingerprint_dict=fingerprint_emb_dict,
            prot_text_emb_dict=prot_text_emb_dict,
            prot_pooling_emb_dict=esm2_pooling_dict,
            device=device
        ).to(device)

    def fit(self, train_loader, NUM_EPOCHS=300):
        print("--- Training GraphMol ---")
        self._train_single_model(
            self.model_graphmol, train_loader, 10, NUM_EPOCHS, f'temp/graphmol_{self.dataset_opt}.pth'
        )

        print("--- Training PreMol ---")
        self._train_single_model(
            self.model_premol, train_loader, 10, NUM_EPOCHS, f'temp/premol_{self.dataset_opt}.pth'
        )

        print("--- Training LLM ---")
        self._train_single_model(
            self.model_llm, train_loader, 20, NUM_EPOCHS, f'temp/llm_{self.dataset_opt}.pth'
        )

        print("--- Training Pooling ---")
        self._train_single_model(
            self.model_pooling, train_loader, 20, NUM_EPOCHS, f'temp/pooling_{self.dataset_opt}.pth'
        )

    def pred(self, test_loader):
        true_label, pred_label_graphmol = self._pred_single_model(self.model_graphmol, test_loader)
        _, pred_label_premol = self._pred_single_model(self.model_premol, test_loader)
        _, pred_label_llm = self._pred_single_model(self.model_llm, test_loader)
        _, pred_label_pooling = self._pred_single_model(self.model_pooling, test_loader)

        # Average predictions for soft voting
        pred_label = (pred_label_graphmol + pred_label_llm + pred_label_premol + pred_label_pooling) / 4.0

        return true_label, pred_label

    def _train_single_model(self, model, train_loader, NUM_PATIENCE, NUM_EPOCHS, temp_path):
        os.makedirs(os.path.dirname(temp_path) or '.', exist_ok=True)
        device = self.device

        optimizer = Adam(model.parameters(), lr=1e-4)
        if self.task == 'dta':
            criterion = CauchyLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode='min', factor=0.98, patience=4, min_lr=1e-5, verbose=True
            )
            early_stopping = EarlyStopping(patience=NUM_PATIENCE, mode='min', delta=1e-3, verbose=True)
        elif self.task == 'dti':
            criterion = BinaryFocalLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode='max', factor=0.98, patience=4, min_lr=1e-5, verbose=True
            )
            early_stopping = EarlyStopping(patience=NUM_PATIENCE, mode='max', delta=1e-3, verbose=True)
        else:
            criterion = BinaryFocalLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode='max', factor=0.98, patience=4, min_lr=1e-5, verbose=True
            )
            early_stopping = EarlyStopping(patience=NUM_PATIENCE, mode='max', delta=1e-5, verbose=True)

        best_model_weights = copy.deepcopy(model.state_dict())

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_train_loss = 0
            all_true_val = []
            all_pred_val = []
            with tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]") as t:
                for i, data in enumerate(t):
                    optimizer.zero_grad()
                    prediction, _, _ = model(data.to(device))
                    loss = criterion(prediction, data.y.view(-1, 1).float().to(device))
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.detach().item()
                    t.set_postfix(Loss=loss.item(), AvgLoss=(total_train_loss / (i + 1)))

                    if self.task != 'dta':
                        probs = torch.sigmoid(prediction.detach().clone())
                        all_true_val.append(data.y.view(-1, 1).cpu())
                        all_pred_val.append(probs.cpu())

            if self.task == 'dta':
                val_metric = total_train_loss / len(train_loader)
                print(f"Epoch {epoch + 1} Validation Loss: {val_metric:.4f}")
            else:
                total_true = torch.cat(all_true_val, dim=0).numpy().flatten()
                total_probs = torch.cat(all_pred_val, dim=0).numpy().flatten()
                val_metric = average_precision_score(total_true, total_probs)
                print(f"Epoch {epoch + 1} Validation AUC: {val_metric:.4f}")

            scheduler.step(val_metric)
            early_stop, best_is_update = early_stopping(val_metric)

            if best_is_update:
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(best_model_weights, temp_path)

            if early_stop:
                print("Early stopping triggered. Training stopped.")
                break

        print("Loading the best model weights for this sub-model...")
        model.load_state_dict(best_model_weights)

    def _pred_single_model(self, model, test_loader):
        device = self.device
        model.eval()
        all_true = []
        all_pred = []

        with torch.no_grad():
            for data in test_loader:
                prediction, _, _ = model(data.to(device))
                all_true.append(data.y.view(-1, 1).cpu())

                if self.task == 'dta':
                    all_pred.append(prediction.cpu())
                else:
                    probs = torch.sigmoid(prediction)
                    all_pred.append(probs.cpu())

        total_true = torch.cat(all_true, dim=0).numpy().flatten()
        total_predict = torch.cat(all_pred, dim=0).numpy().flatten()
        return total_true, total_predict
