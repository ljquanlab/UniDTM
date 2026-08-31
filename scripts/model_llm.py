import torch
import torch.nn as nn
import torch.nn.functional as F


def get_prot_llm_batch_embeddings(prot_seq_batch, emb_dict):
    batch_tensors = [torch.tensor(emb_dict.get(seq)) for seq in prot_seq_batch]
    return torch.stack(batch_tensors)


def get_drug_llm_batch_embeddings(smiles_batch, emb_dict):
    batch_tensors = [torch.tensor(emb_dict.get(s)) for s in smiles_batch]
    return torch.stack(batch_tensors)


class TransformerBindingInteraction(nn.Module):
    def __init__(self, dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.dim = dim
        self.binding_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, drug_feat, prot_feat):
        batch_size = drug_feat.size(0)
        d_seq = drug_feat.unsqueeze(1)
        p_seq = prot_feat.unsqueeze(1)
        b_seq = self.binding_token.expand(batch_size, -1, -1)

        combined_seq = torch.cat([b_seq, d_seq, p_seq], dim=1)
        attended_seq = self.transformer(combined_seq)

        binding_feat = attended_seq[:, 0, :]
        return self.norm(binding_feat)


class ModelLLM(torch.nn.Module):
    def __init__(self, llm_emb_dict, prot_text_emb_dict, device, hidden_dim=128):
        super(ModelLLM, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.dropout = 0.3

        self.llm_emb_dict = llm_emb_dict
        self.prot_text_emb_dict = prot_text_emb_dict

        # Minimal projection: single layer to preserve raw semantics; input Dropout as data augmentation
        self.protein_text_projection = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(1024, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )

        self.drug_text_projection = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(1024, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )

        self.interaction_layer = TransformerBindingInteraction(
            dim=self.hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=self.dropout
        )

        # Input = Binding(128) + cosine similarity(1) + Euclidean distance(1) + projected concat(256)
        concat_dim = self.hidden_dim + 2 + (self.hidden_dim * 2)

        self.predictor = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.BatchNorm1d(256),  # BN is better than LN at preventing deep overfitting
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        prot_raw_feat = get_prot_llm_batch_embeddings(
            data.protein_sequence, self.prot_text_emb_dict
        ).to(self.device)
        drug_raw_feat = get_drug_llm_batch_embeddings(
            data.smiles, self.llm_emb_dict
        ).to(self.device)

        # Compute similarity in the original 1024-d space (highly informative for LLM embeddings)
        raw_cos_sim = F.cosine_similarity(drug_raw_feat, prot_raw_feat, dim=-1).unsqueeze(-1)
        raw_l2_dist = torch.norm(drug_raw_feat - prot_raw_feat, p=2, dim=-1).unsqueeze(-1)

        prot_feat = self.protein_text_projection(prot_raw_feat)
        drug_feat = self.drug_text_projection(drug_raw_feat)

        fused_binding_feat = self.interaction_layer(drug_feat, prot_feat)

        # Residual concat: Binding + raw similarity/distance + projected individual features
        # Individual features help prevent the Transformer from washing out useful signals
        final_feat = torch.cat([
            fused_binding_feat,
            raw_cos_sim,
            raw_l2_dist,
            drug_feat,
            prot_feat
        ], dim=-1)

        prediction = self.predictor(final_feat)

        return prediction, None, None
