import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class MultimodalLocalCrossAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Independent saliency gate: drug decides which tokens are chemically active
        self.drug_saliency = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Protein decides which residues may form binding pockets
        self.prot_saliency = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Bidirectional cross-attention
        self.drug_to_prot_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.prot_to_drug_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.layer_norm_d = nn.LayerNorm(hidden_dim)
        self.layer_norm_p = nn.LayerNorm(hidden_dim)

    def forward(self, drug_feat, prot_feat, drug_mask, prot_mask):
        """
        drug_feat: [Batch, Max_Tokens, Dim]
        prot_feat: [Batch, Max_Residues, Dim]
        drug_mask/prot_mask: [Batch, SeqLen], True = invalid positions
        """
        # A. Independent saliency filtering of noise
        drug_gate_logits = self.drug_saliency(drug_feat)
        prot_gate_logits = self.prot_saliency(prot_feat)

        # Set padding logits to a large negative so Sigmoid yields ~0
        drug_gate_logits = drug_gate_logits.masked_fill(drug_mask.unsqueeze(-1), -1e9)
        prot_gate_logits = prot_gate_logits.masked_fill(prot_mask.unsqueeze(-1), -1e9)

        drug_gate = torch.sigmoid(drug_gate_logits)
        prot_gate = torch.sigmoid(prot_gate_logits)

        # Background residues are suppressed; likely pockets are amplified
        drug_feat_gated = drug_feat * drug_gate
        prot_feat_gated = prot_feat * prot_gate

        # B. Cross-attention (on filtered features)
        d_attended, _ = self.drug_to_prot_attn(
            query=drug_feat_gated, key=prot_feat_gated, value=prot_feat_gated,
            key_padding_mask=prot_mask
        )
        d_out = self.layer_norm_d(drug_feat_gated + d_attended)

        p_attended, _ = self.prot_to_drug_attn(
            query=prot_feat_gated, key=drug_feat_gated, value=drug_feat_gated,
            key_padding_mask=drug_mask
        )
        p_out = self.layer_norm_p(prot_feat_gated + p_attended)

        # C. Dynamic pooling: gate-weighted average focusing on important local features
        d_gate_sum = drug_gate.sum(dim=1).clamp(min=1e-9)
        p_gate_sum = prot_gate.sum(dim=1).clamp(min=1e-9)

        d_global = (d_out * drug_gate).sum(dim=1) / d_gate_sum
        p_global = (p_out * prot_gate).sum(dim=1) / p_gate_sum

        # Additional max pooling to capture extreme binding signals
        d_out_max = d_out.masked_fill(drug_mask.unsqueeze(-1), -1e9)
        p_out_max = p_out.masked_fill(prot_mask.unsqueeze(-1), -1e9)

        d_max = d_out_max.max(dim=1)[0]
        p_max = p_out_max.max(dim=1)[0]

        return d_global + d_max, p_global + p_max


class ModelPreMol(torch.nn.Module):
    def __init__(self, esm2_dict, chemberta_dict, device, hidden_dim=128):
        super(ModelPreMol, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.dropout = 0.3

        self.esm2_dict = esm2_dict
        self.chemberta_dict = chemberta_dict

        # Protein sequence local projection (1280 -> 128)
        self.esm_projection = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.hidden_dim)
        )

        # Drug sequence local projection (ChemBERTa 384 -> 128)
        self.chemberta_projection = nn.Sequential(
            nn.Linear(384, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.hidden_dim)
        )

        self.local_interaction = MultimodalLocalCrossAttention(
            hidden_dim=self.hidden_dim, dropout=self.dropout
        )

        # Input is inter_prod + inter_diff (hidden_dim * 2)
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 512),
            nn.BatchNorm1d(512),  # Stabilize feature distribution for unseen samples
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        # A. Protein local features (ESM2)
        esm_list = []
        for seq in data.protein_sequence:
            emb = self.esm2_dict.get(seq).to(self.device)
            esm_list.append(emb)

        padded_esm = pad_sequence(esm_list, batch_first=True, padding_value=0.0)
        esm_mask = (padded_esm.sum(dim=-1) == 0)
        prot_local_feat = self.esm_projection(padded_esm)

        # B. Drug local features (ChemBERTa)
        chembert_list = []
        for smi in data.smiles:
            emb = self.chemberta_dict.get(smi).to(self.device)
            chembert_list.append(emb)

        padded_chemberta = pad_sequence(chembert_list, batch_first=True, padding_value=0.0)
        chemberta_mask = (padded_chemberta.sum(dim=-1) == 0)
        drug_local_feat = self.chemberta_projection(padded_chemberta)

        # C. Local cross-interaction
        d_global, p_global = self.local_interaction(
            drug_local_feat, prot_local_feat,
            drug_mask=chemberta_mask, prot_mask=esm_mask
        )

        # D. Heuristic interaction matching: element-wise product + absolute difference
        inter_prod = d_global * p_global
        inter_diff = torch.abs(d_global - p_global)

        final_representation = torch.cat([inter_prod, inter_diff], dim=-1)
        Prediction = self.predictor(final_representation)

        return Prediction, None, None
