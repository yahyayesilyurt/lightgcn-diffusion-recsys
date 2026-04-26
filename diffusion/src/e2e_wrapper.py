import torch
import torch.nn as nn
import torch.nn.functional as F

from src.domain_specific_aggregator import DomainSpecificAggregator
from src.attention_condition_generator import AttentionConditionGenerator


class E2EWrapper(nn.Module):
    """
    End-to-end wrapper for Triple-Stream Architecture.

    Domain mapping:
      Source domain : User's interaction history in source domain -> h_u_source
      Target domain : Predict next interaction in target domain -> h_u_target

    Triple-Stream:
      Stream 1 — GNN Anchor    : h_u_cross  (GAT user embedding)
      Stream 2 — Source Intent : h_u_source (Source sequence aggregation)
      Stream 3 — Target Intent : h_u_target (Target sequence aggregation)

    Ablation flag:
      use_source_stream=False -> h_u_source = h_u_cross (source stream disabled)

    Embedding strategy:
      All embeddings freeze=True — Topological structure from GAT must be preserved.

    Index conventions:
      user_ids    : 0-indexed -> +1 offset applied internally
      source_seq  : 1-indexed -> 0 = padding token
      target_seq  : 1-indexed -> 0 = padding token
    """

    def __init__(
        self,
        padded_user_embs,
        padded_source_embs,    # Source domain embeddings
        padded_target_embs,    # Target domain embeddings
        embed_dim=128,
        num_heads=4,
        dropout=0.1,
        use_source_stream=True
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
        )

        self.use_source_stream = use_source_stream

        # --- GNN Embeddings (frozen) ---
        self.user_embedding = nn.Embedding.from_pretrained(
            padded_user_embs, freeze=True, padding_idx=0
        )
        self.target_embedding = nn.Embedding.from_pretrained(
            padded_target_embs, freeze=True, padding_idx=0
        )
        self.source_embedding = nn.Embedding.from_pretrained(
            padded_source_embs, freeze=True, padding_idx=0
        )

        # --- Projection layers ---
        self.user_proj   = self._make_proj(embed_dim, dropout)
        self.target_proj = self._make_proj(embed_dim, dropout)

        if self.use_source_stream:
            self.source_proj = self._make_proj(embed_dim, dropout)
            self.source_aggregator = DomainSpecificAggregator(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=embed_dim * 4,
                dropout=dropout
            )

        # --- Target Aggregator (Target sequence) ---
        self.target_aggregator = DomainSpecificAggregator(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=embed_dim * 4,
            dropout=dropout
        )

        # --- GNN-Anchored Condition Generator ---
        self.condition_generator = AttentionConditionGenerator(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=embed_dim * 4,
            dropout=dropout
        )

    @staticmethod
    def _make_proj(embed_dim: int, dropout: float) -> nn.Sequential:
        """
        Projection block: normalize raw GAT embedding -> expand -> compress.
        """
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(
        self,
        user_ids,
        target_seq_ids,
        target_mask,
        source_seq_ids=None,
        source_mask=None,
        return_fusion_weights=False,
    ):
        """
        Args:
            user_ids             : (B,)   — 0-indexed user IDs
            target_seq_ids       : (B, S) — 1-indexed target history, 0=padding
            target_mask          : (B, S) — True where padding
            source_seq_ids       : (B, S) — 1-indexed source history, 0=padding
                                            (Not used if use_source_stream=False)
            source_mask          : (B, S) — True where padding
                                            (Not used if use_source_stream=False)
            return_fusion_weights: If True, return cross-attention weights
                (source vs target slots) before final L2 normalize.

        Returns:
            c_ud : (B, D) — L2-normalized diffusion condition vector
            Or (c_ud, fusion_weights) when return_fusion_weights=True;
            fusion_weights (B, 2): [:,0] source, [:,1] target (sums to 1).
        """

        # ── Stream 1: GNN Anchor ─────────────────────────────────────────
        h_u_cross = self.user_proj(
            self.user_embedding(user_ids + 1)
        )                                                                 # (B, D)

        # ── Stream 2: Short-Term Target Intent ───────────────────────────
        target_empty     = target_mask.all(dim=1)
        safe_target_mask = target_mask.clone()
        safe_target_mask[target_empty, 0] = False

        h_u_target = self.target_aggregator(
            self.target_proj(self.target_embedding(target_seq_ids)),
            key_padding_mask=safe_target_mask
        )                                                                 # (B, D)
        h_u_target = torch.where(
            target_empty.unsqueeze(1), h_u_cross, h_u_target
        )

        # ── Stream 3: Long-Term Source Intent ────────────────────────────
        if self.use_source_stream:
            assert source_seq_ids is not None and source_mask is not None, (
                "source_seq_ids and source_mask required when use_source_stream=True"
            )
            source_empty     = source_mask.all(dim=1)
            safe_source_mask = source_mask.clone()
            safe_source_mask[source_empty, 0] = False

            h_u_source = self.source_aggregator(
                self.source_proj(self.source_embedding(source_seq_ids)),
                key_padding_mask=safe_source_mask
            )                                                             # (B, D)
            h_u_source = torch.where(
                source_empty.unsqueeze(1), h_u_cross, h_u_source
            )
        else:
            # Ablation: No source stream -> GNN anchor serves as its own source
            h_u_source = h_u_cross                                        # (B, D)

        # ── Triple-Stream Fusion ─────────────────────────────────────────
        if return_fusion_weights:
            c_ud_raw, fusion_weights = self.condition_generator(
                h_u_cross=h_u_cross,
                h_u_source=h_u_source,
                h_u_target=h_u_target,
                return_attention=True,
            )                                                             # (B, D), (B, 2)
            c_ud = F.normalize(c_ud_raw, p=2, dim=1)
            return c_ud, fusion_weights

        c_ud = self.condition_generator(
            h_u_cross=h_u_cross,
            h_u_source=h_u_source,
            h_u_target=h_u_target,
        )                                                                 # (B, D)

        return F.normalize(c_ud, p=2, dim=1)