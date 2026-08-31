import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_max_pool
import math


def get_prot_llm_batch_embeddings(prot_seq_batch, emb_dict):
    batch_tensors = [torch.tensor(emb_dict.get(seq)) for seq in prot_seq_batch]
    return torch.stack(batch_tensors)


def get_llm_batch_embeddings(smiles_batch, emb_dict):
    batch_tensors = [torch.tensor(emb_dict.get(s)) for s in smiles_batch]
    return torch.stack(batch_tensors)


def get_prot_esm2_pooling_batch_embeddings(prot_seq_batch, emb_dict):
    batch_tensors = [emb_dict.get(seq) for seq in prot_seq_batch]
    return torch.stack(batch_tensors).squeeze(1)


def get_chemberta_batch_embeddings(smiles_batch, chemberta_emb_dict):
    # ChemBERTa pooling dim = 384
    batch_tensors = [chemberta_emb_dict.get(s) for s in smiles_batch]
    return torch.stack(batch_tensors)


def get_smiles_fingerprint_embeddings(smiles_batch, fingerprint_dict):
    # Morgan fingerprint dim = 2048
    batch_tensors = [fingerprint_dict.get(s).float() for s in smiles_batch]
    return torch.stack(batch_tensors)


class ChembertaProjection(nn.Module):
    def __init__(self, input_dim=384, dropout=0.1, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, precomputed_features):
        """
        Args:
            precomputed_features: (batch_size, 384) Tensor looked up from the dictionary
        """
        return self.projection(precomputed_features)


class ImprovedEncoder(nn.Module):
    def __init__(self, node_in=94, edge_in=11, hidden_dim=256, Final_dim=128, dropout=0.2):
        super().__init__()

        # Feature pre-projection: map sparse one-hot node/edge features to continuous space
        self.node_emb = nn.Linear(node_in, hidden_dim)
        self.edge_emb = nn.Linear(edge_in, hidden_dim)

        # Two GINE layers covering local molecular substructures
        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=hidden_dim
        )
        self.conv2 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=hidden_dim
        )

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Concat two max-pool layers then project to Final_dim
        self.Drug_FCs = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, Final_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = self.ln1(x1)
        x1 = F.gelu(x1)

        # Second layer + residual to prevent degradation
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.ln2(x2)
        x2 = F.gelu(x2) + x1

        pool1 = global_max_pool(x1, batch)
        pool2 = global_max_pool(x2, batch)
        x_combined = torch.cat([pool1, pool2], dim=1)

        return self.Drug_FCs(x_combined)


class AffinityRegressionHead(nn.Module):
    def __init__(self, drug_dim, protein_dim, n_output=1, dropout=0.2):
        super(AffinityRegressionHead, self).__init__()

        # Dimensionality alignment: project drug and protein into a shared hidden space
        self.hidden_dim = 512
        self.drug_project = nn.Linear(drug_dim, self.hidden_dim)
        self.prot_project = nn.Linear(protein_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)

        # Input contains [drug, protein, product, difference] to characterize matching from multiple angles
        combined_dim = self.hidden_dim * 4

        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, n_output)
        )

    def forward(self, drug_feat, protein_feat):
        d = self.ln(F.gelu(self.drug_project(drug_feat)))
        p = self.ln(F.gelu(self.prot_project(protein_feat)))

        inter_prod = d * p
        inter_diff = torch.abs(d - p)
        combined = torch.cat([d, p, inter_prod, inter_diff], dim=-1)

        return self.regressor(combined)


class TargetAwareRouter(nn.Module):
    """Target-aware weighted fusion of drug modalities, using the protein as Query."""

    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

        self.norm_graph = nn.LayerNorm(dim)
        self.norm_chem = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(dim)
        self.norm_fp = nn.LayerNorm(dim)
        self.norm_prot = nn.LayerNorm(dim)

        # Protein as Query; drug modalities as Keys
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)

        # Nonlinear complementary scoring for stronger expressiveness
        self.nonlinear_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, graph_feat, seq_feat, text_feat, fingerprint_feat, protein_feat):
        graph_feat = self.norm_graph(graph_feat)
        seq_feat = self.norm_chem(seq_feat)
        text_feat = self.norm_text(text_feat)
        fingerprint_feat = self.norm_fp(fingerprint_feat)
        protein_feat = self.norm_prot(protein_feat)

        Q = self.proj_q(protein_feat)
        K_stack = torch.stack([graph_feat, seq_feat, text_feat, fingerprint_feat], dim=1)
        K = self.proj_k(K_stack)

        # Path A: scaled dot-product attention capturing directional alignment
        attn_scores = torch.bmm(Q.unsqueeze(1), K.transpose(1, 2)).squeeze(1) / math.sqrt(self.dim)

        # Path B: nonlinear context scoring capturing complex dependencies
        Q_expanded = protein_feat.unsqueeze(1).expand(-1, 4, -1)
        concat_context = torch.cat([K_stack, Q_expanded], dim=-1)
        mlp_scores = self.nonlinear_scorer(concat_context).squeeze(-1)

        combined_logits = attn_scores + mlp_scores

        # Temperature T=1.5 to reduce weight polarization
        T = 1.5
        weights = F.softmax(combined_logits / T, dim=-1)

        w_graph = weights[:, 0].unsqueeze(1)
        w_seq = weights[:, 1].unsqueeze(1)
        w_text = weights[:, 2].unsqueeze(1)
        w_fp = weights[:, 3].unsqueeze(1)

        fused_drug_feat = (
            w_graph * graph_feat + w_seq * seq_feat + w_text * text_feat + w_fp * fingerprint_feat
        )

        return fused_drug_feat, weights


class DrugAwareProtRouter(nn.Module):
    """Drug-context-aware weighted fusion of protein modalities, using drug context as Query."""

    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

        self.norm_seq = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(dim)
        self.norm_drug = nn.LayerNorm(dim)

        # Drug as Query; protein modalities as Keys
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)

        self.nonlinear_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, prot_seq_feat, prot_text_feat, drug_context):
        p_seq = self.norm_seq(prot_seq_feat)
        p_text = self.norm_text(prot_text_feat)
        d_ctx = self.norm_drug(drug_context)

        Q = self.proj_q(d_ctx)
        K_stack = torch.stack([p_seq, p_text], dim=1)
        K = self.proj_k(K_stack)

        attn_scores = torch.bmm(Q.unsqueeze(1), K.transpose(1, 2)).squeeze(1) / math.sqrt(self.dim)

        Q_expanded = d_ctx.unsqueeze(1).expand(-1, 2, -1)
        concat_context = torch.cat([K_stack, Q_expanded], dim=-1)
        mlp_scores = self.nonlinear_scorer(concat_context).squeeze(-1)

        combined_logits = attn_scores + mlp_scores

        T = 1.5
        weights = F.softmax(combined_logits / T, dim=-1)

        w_seq = weights[:, 0].unsqueeze(1)
        w_text = weights[:, 1].unsqueeze(1)
        fused_prot_feat = (w_seq * p_seq) + (w_text * p_text)

        return fused_prot_feat, weights


class ModelPooling(torch.nn.Module):
    def __init__(self, chemberta_emb_dict, llm_emb_dict, fingerprint_dict, prot_text_emb_dict, prot_pooling_emb_dict, device):
        super(ModelPooling, self).__init__()
        self.output_dim = 128
        self.device = device
        self.dropout = 0.3

        self.chemberta_emb_dict = chemberta_emb_dict
        self.llm_emb_dict = llm_emb_dict
        self.fingerprint_dict = fingerprint_dict
        self.prot_text_emb_dict = prot_text_emb_dict
        self.prot_pooling_emb_dict = prot_pooling_emb_dict

        self.encoder = ImprovedEncoder()

        self.chem_projection = ChembertaProjection(dropout=self.dropout, output_dim=self.output_dim)
        self.chem_residual = nn.Linear(384, self.output_dim)

        self.fp_proj = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128)
        )
        self.fp_residual = nn.Linear(2048, self.output_dim)

        self.modality_router = TargetAwareRouter(dim=self.output_dim)

        esm_feature_dim = 1280
        adapter_hidden_dim = 512
        final_dim = 128

        # ESM2 pooling adapter
        self.adapter = nn.Sequential(
            nn.Linear(esm_feature_dim, adapter_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(adapter_hidden_dim, adapter_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(adapter_hidden_dim // 2, final_dim)
        )
        self.residual = nn.Linear(esm_feature_dim, final_dim)

        self.fc = AffinityRegressionHead(
            drug_dim=self.output_dim, protein_dim=final_dim, n_output=1, dropout=self.dropout
        )

        llm_dim = 1024
        self.llm_projection = nn.Sequential(
            nn.Linear(llm_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, final_dim)
        )
        self.llm_residual = nn.Linear(llm_dim, final_dim)

        self.llm_prot_projection = nn.Sequential(
            nn.Linear(llm_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, final_dim)
        )
        self.llm_prot_residual = nn.Linear(llm_dim, final_dim)
        self.prot_router = DrugAwareProtRouter(dim=final_dim)

    def forward(self, data):
        # Protein ESM2 pooling + residual adapter
        Protein_vector = get_prot_esm2_pooling_batch_embeddings(
            data.protein_sequence, self.prot_pooling_emb_dict
        ).to(self.device)
        Protein_vector_copy = Protein_vector.clone()
        adapted_features = self.adapter(Protein_vector)
        residual_features = self.residual(Protein_vector_copy)
        prot_seq_feat = adapted_features + residual_features

        # Protein LLM text features
        prot_llm_feat = get_prot_llm_batch_embeddings(
            data.protein_sequence, self.prot_text_emb_dict
        ).to(self.device)
        prot_llm_feat_copy = prot_llm_feat.clone()
        prot_llm_feat = self.llm_prot_projection(prot_llm_feat) + self.llm_prot_residual(prot_llm_feat_copy)

        # Drug graph structural features
        PMVO = self.encoder(data)

        # ChemBERTa pooling + residual
        chem_feat = get_chemberta_batch_embeddings(data.smiles, self.chemberta_emb_dict).to(self.device)
        chem_feat_copy = chem_feat.clone()
        CHEM = self.chem_projection(chem_feat) + self.chem_residual(chem_feat_copy)

        # Drug LLM text features + residual
        text_feat = get_llm_batch_embeddings(data.smiles, self.llm_emb_dict).to(self.device)
        text_feat_copy = text_feat.clone()
        text_feat = self.llm_projection(text_feat) + self.llm_residual(text_feat_copy)

        # Fingerprint features + residual
        fingerprint_emb = get_smiles_fingerprint_embeddings(
            data.smiles, self.fingerprint_dict
        ).to(self.device)
        fingerprint_emb_copy = fingerprint_emb.clone()
        fingerprint_emb = self.fp_proj(fingerprint_emb) + self.fp_residual(fingerprint_emb_copy)

        # Bidirectional routing: coarse drug context guides protein fusion, then protein guides drug fusion
        drug_context = (PMVO + CHEM + text_feat + fingerprint_emb) / 4.0
        Fused_Protein, w_prot = self.prot_router(prot_seq_feat, prot_llm_feat, drug_context)

        Protein_vector = (prot_seq_feat + prot_llm_feat) / 2.0
        Fused_Drug, w_drug = self.modality_router(PMVO, CHEM, text_feat, fingerprint_emb, Protein_vector)

        Prediction = self.fc(Fused_Drug, Fused_Protein)

        return Prediction, w_drug, w_prot
