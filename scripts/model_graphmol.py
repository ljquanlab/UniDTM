'''
Use only the drug graph for atom-level features and ESM2 pretrained residue-level features
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch
from torch.nn.utils.rnn import pad_sequence


class ImprovedEncoder(nn.Module):
    def __init__(self, node_in=94, edge_in=11, hidden_dim=256, Final_dim=128, dropout=0.2):
        super().__init__()

        # Feature pre-projection
        self.node_emb = nn.Linear(node_in, hidden_dim)
        self.edge_emb = nn.Linear(edge_in, hidden_dim)

        # Three GINE layers covering 5-6 membered rings
        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=hidden_dim
        )
        self.conv2 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=hidden_dim
        )
        self.conv3 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=hidden_dim
        )

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # Node-level Jumping Knowledge: concat 3 layer node features then project
        self.node_FCs = nn.Sequential(
            nn.Linear(hidden_dim * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, Final_dim)
        )

    def forward(self, data):
        # x: [Total_nodes, node_in]
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = self.ln1(x1)
        x1 = F.gelu(x1)

        # Second layer + residual
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.ln2(x2)
        x2 = F.gelu(x2) + x1

        # Third layer + residual
        x3 = self.conv3(x2, edge_index, edge_attr)
        x3 = self.ln3(x3)
        x3 = F.gelu(x3) + x2

        # Node-level Jumping Knowledge, preserving topology
        # [Total_nodes, hidden_dim * 3]
        x_node_jk = torch.cat([x1, x2, x3], dim=-1)
        # [Total_nodes, Final_dim]
        node_features = self.node_FCs(x_node_jk)

        # Convert to dense matrix for subsequent Cross-Attention
        # dense_x: [Batch_size, Max_nodes_in_batch, Final_dim]
        # node_mask: True = real atom, False = padding
        dense_x, node_mask = to_dense_batch(node_features, batch)

        # In nn.MultiheadAttention key_padding_mask, True means positions to ignore (padding)
        padding_mask = ~node_mask

        return dense_x, padding_mask


class MultimodalLocalCrossAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Drug (Query) attends to protein (Key/Value)
        self.drug_to_prot_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        # Protein (Query) attends to drug (Key/Value)
        self.prot_to_drug_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.layer_norm_d = nn.LayerNorm(hidden_dim)
        self.layer_norm_p = nn.LayerNorm(hidden_dim)

    def forward(self, drug_feat, prot_feat, drug_mask, prot_mask):
        """
        drug_feat: [Batch, Max_Nodes, Dim]
        prot_feat: [Batch, Max_Residues, Dim]
        drug_mask/prot_mask: [Batch, SeqLen], True = padded invalid positions
        """
        d_attended, _ = self.drug_to_prot_attn(
            query=drug_feat, key=prot_feat, value=prot_feat,
            key_padding_mask=prot_mask  # drug ignores protein padding
        )
        d_out = self.layer_norm_d(drug_feat + d_attended)

        p_attended, _ = self.prot_to_drug_attn(
            query=prot_feat, key=drug_feat, value=drug_feat,
            key_padding_mask=drug_mask  # protein ignores drug padding
        )
        p_out = self.layer_norm_p(prot_feat + p_attended)

        # Masked global mean pooling
        d_out = d_out.masked_fill(drug_mask.unsqueeze(-1), 0.0)
        p_out = p_out.masked_fill(prot_mask.unsqueeze(-1), 0.0)

        # Use valid length as denominator to avoid division by zero
        d_valid_len = (~drug_mask).sum(dim=1, keepdim=True).clamp(min=1e-9)
        p_valid_len = (~prot_mask).sum(dim=1, keepdim=True).clamp(min=1e-9)

        d_global = d_out.sum(dim=1) / d_valid_len
        p_global = p_out.sum(dim=1) / p_valid_len

        return d_global, p_global


class ModelGraphMol(torch.nn.Module):
    def __init__(self, esm2_dict, device, hidden_dim=128):
        super(ModelGraphMol, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.dropout = 0.3

        self.esm2_dict = esm2_dict  # ESM2 residue-level feature dictionary

        self.encoder = ImprovedEncoder(Final_dim=self.hidden_dim, dropout=self.dropout)

        # Protein sequence local projection: ESM 1280-d -> unified 128-d
        self.esm_projection = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(1280, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )

        self.local_interaction = MultimodalLocalCrossAttention(
            hidden_dim=self.hidden_dim, dropout=self.dropout
        )

        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        # A. Protein local features (ESM2)
        esm_list = []
        for seq in data.protein_sequence:
            emb = self.esm2_dict.get(seq).to(self.device)
            esm_list.append(emb)

        # Dynamic padding: [Batch, Max_Residues, 1280]
        padded_esm = pad_sequence(esm_list, batch_first=True, padding_value=0.0)
        # All-zero positions are treated as padding (True)
        esm_mask = (padded_esm.sum(dim=-1) == 0)
        prot_local_feat = self.esm_projection(padded_esm)

        # B. Drug local features (GNN)
        # drug_local_feat: [Batch, Max_Nodes, 128], gnn_mask: [Batch, Max_Nodes]
        drug_local_feat, gnn_mask = self.encoder(data)

        # C. Local cross-interaction then safe pooling
        d_global_struct, p_global_struct = self.local_interaction(
            drug_local_feat, prot_local_feat,
            drug_mask=gnn_mask, prot_mask=esm_mask
        )

        # Element-wise product: co-activated feature dimensions
        inter_prod = d_global_struct * p_global_struct
        # Absolute difference: complementary or opposing feature dimensions
        inter_diff = torch.abs(d_global_struct - p_global_struct)

        final_representation = torch.cat([inter_prod, inter_diff], dim=-1)
        Prediction = self.predictor(final_representation)

        return Prediction, None, None
