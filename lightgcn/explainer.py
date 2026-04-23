"""
Post-hoc explainability for LightGCN: scalar contribution weights per graph node.

Same linear decomposition as the reference LightGCN project (see LightGCN/explain.py):

    E_u_final = (1/(L+1)) * [I + A + A^2 + ... + A^L]_u  ·  E^(0)

Weights w(u, j) depend only on the normalized adjacency A and L, not on learned vectors.

The training adjacency from graph_utils.create_adj_matrix must be converted to SciPy
for efficient repeated sparse mat-vecs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

if TYPE_CHECKING:
    from models.lightgcn import LightGCN


def sparse_torch_adj_to_scipy_csr(adj: Tensor) -> sp.csr_matrix:
    """
    Convert a PyTorch sparse COO adjacency (CPU or CUDA) to SciPy CSR on CPU.

    Expects a coalescible COO; shape (n_nodes, n_nodes).
    """
    if not adj.is_sparse:
        raise TypeError("adj must be a torch sparse tensor")
    a = adj.coalesce()
    if a.device.type != "cpu":
        a = a.cpu()
    idx = a.indices()
    val = a.values()
    n = a.shape[0]
    r = idx[0].numpy()
    c = idx[1].numpy()
    data = val.double().numpy()
    return sp.coo_matrix((data, (r, c)), shape=(n, n), dtype=np.float64).tocsr()


class LightGCNExplainer:
    """
    Explains a user's final LightGCN embedding via exact scalar weights on
    every node's initial (layer-0) embedding. Matches training when A and L
    are the same as the forward pass.
    """

    def __init__(
        self,
        adj_mat: sp.spmatrix,
        n_users: int,
        n_items: int,
        n_layers: int,
        train_items: Optional[Dict[int, List[int]]] = None,
        domain_map: Optional[Dict[int, str]] = None,
    ) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.n_nodes = n_users + n_items
        self.train_items: Dict[int, List[int]] = train_items or {}
        self.domain_map = domain_map

        if adj_mat.shape != (self.n_nodes, self.n_nodes):
            raise ValueError(
                f"adj_mat shape {adj_mat.shape} does not match "
                f"expected ({self.n_nodes}, {self.n_nodes})"
            )

        self.A_T = adj_mat.T.tocsr()

    def _compute_weight_vector(self, user_id: int) -> np.ndarray:
        v = np.zeros(self.n_nodes, dtype=np.float64)
        v[user_id] = 1.0
        total = v.copy()
        for _ in range(self.n_layers):
            v = self.A_T.dot(v)
            total += v
        total /= self.n_layers + 1
        return total

    def _compute_layerwise_vectors(self, user_id: int) -> List[np.ndarray]:
        v = np.zeros(self.n_nodes, dtype=np.float64)
        v[user_id] = 1.0
        layers = [v.copy()]
        for _ in range(self.n_layers):
            v = self.A_T.dot(v)
            layers.append(v.copy())
        return layers

    def explain_user(self, user_id: int, top_k: int = 20) -> Dict[str, Any]:
        if user_id < 0 or user_id >= self.n_users:
            raise ValueError(
                f"user_id {user_id} out of range [0, {self.n_users})"
            )

        total = self._compute_weight_vector(user_id)
        weight_sum = float(total.sum())
        direct_items = set(self.train_items.get(user_id, []))

        self_weight = float(total[user_id])
        self_pct = (self_weight / weight_sum * 100) if weight_sum else 0.0

        item_weights = total[self.n_users :]
        top_item_idx = np.argsort(-item_weights)
        item_contributions: List[Tuple[int, float, float, bool]] = []
        for idx in top_item_idx:
            if len(item_contributions) >= top_k:
                break
            w = float(item_weights[idx])
            if w <= 0:
                break
            pct = w / weight_sum * 100 if weight_sum else 0.0
            item_contributions.append(
                (int(idx), w, float(pct), int(idx) in direct_items)
            )

        user_weights = total[: self.n_users].copy()
        user_weights[user_id] = 0.0
        top_user_idx = np.argsort(-user_weights)
        user_contributions: List[Tuple[int, float, float]] = []
        for idx in top_user_idx:
            if len(user_contributions) >= top_k:
                break
            w = float(user_weights[idx])
            if w <= 0:
                break
            pct = w / weight_sum * 100 if weight_sum else 0.0
            user_contributions.append((int(idx), w, float(pct)))

        result: Dict[str, Any] = {
            "user_id": user_id,
            "n_layers": self.n_layers,
            "self_weight": self_weight,
            "self_pct": float(self_pct),
            "item_contributions": item_contributions,
            "user_contributions": user_contributions,
            "total_weight_sum": weight_sum,
        }

        if self.domain_map is not None:
            domain_weights: Dict[str, float] = {}
            for i in range(self.n_items):
                w = float(item_weights[i])
                if w <= 0:
                    continue
                domain = self.domain_map.get(i, "unknown")
                domain_weights[domain] = domain_weights.get(domain, 0.0) + w
            domain_summary = {
                d: (dw / weight_sum * 100) for d, dw in domain_weights.items()
            }
            domain_summary = {
                d: v for d, v in sorted(domain_summary.items(), key=lambda x: -x[1])
            }
            domain_summary["self"] = float(self_pct)
            result["domain_summary"] = domain_summary

        return result

    def explain_user_layerwise(
        self, user_id: int, top_k_per_layer: int = 10
    ) -> List[Dict[str, Any]]:
        if user_id < 0 or user_id >= self.n_users:
            raise ValueError(
                f"user_id {user_id} out of range [0, {self.n_users})"
            )

        layers = self._compute_layerwise_vectors(user_id)
        factor = 1.0 / (self.n_layers + 1)
        direct_items = set(self.train_items.get(user_id, []))

        aggregated = np.zeros(self.n_nodes, dtype=np.float64)
        for lv in layers:
            aggregated += lv
        aggregated *= factor
        weight_sum = float(aggregated.sum())

        results: List[Dict[str, Any]] = []
        for k, layer_v in enumerate(layers):
            item_w = layer_v[self.n_users :]
            user_w = layer_v[: self.n_users].copy()
            user_w[user_id] = 0.0

            top_items: List[Tuple[int, float, float, bool]] = []
            for idx in np.argsort(-item_w):
                if len(top_items) >= top_k_per_layer:
                    break
                w = float(item_w[idx])
                if w <= 0:
                    break
                pct = (w * factor / weight_sum * 100) if weight_sum else 0.0
                top_items.append(
                    (int(idx), w * factor, float(pct), int(idx) in direct_items)
                )

            top_users: List[Tuple[int, float, float]] = []
            for idx in np.argsort(-user_w):
                if len(top_users) >= top_k_per_layer:
                    break
                w = float(user_w[idx])
                if w <= 0:
                    break
                pct = (w * factor / weight_sum * 100) if weight_sum else 0.0
                top_users.append((int(idx), w * factor, float(pct)))

            if k == 0:
                hop_type = "self"
            elif k % 2 == 1:
                hop_type = "items"
            else:
                hop_type = "users"

            entry: Dict[str, Any] = {
                "layer": k,
                "hop_type": hop_type,
                "top_items": top_items,
                "top_users": top_users,
            }

            if k == 0:
                sw = float(layers[0][user_id])
                entry["self_weight"] = sw * factor
                entry["self_pct"] = (
                    float(sw * factor / weight_sum * 100) if weight_sum else 0.0
                )

            results.append(entry)
        return results

    def get_raw_weights(self, user_id: int) -> np.ndarray:
        return self._compute_weight_vector(user_id)


def build_explainer_from_model(
    model: "LightGCN",
    train_interactions: Dict[int, List[int]],
    domain_map: Optional[Dict[int, str]] = None,
) -> LightGCNExplainer:
    """
    Build a LightGCNExplainer from a trained model's adjacency and hyperparameters.

    The model's adjacency must have been set via set_adjacency_matrix (same graph
    as training). Does not call forward; read-only.
    """
    if model.adj is None:
        raise RuntimeError("model.adj is None; call set_adjacency_matrix first")
    adj_scipy = sparse_torch_adj_to_scipy_csr(model.adj)
    return LightGCNExplainer(
        adj_mat=adj_scipy,
        n_users=model.n_users,
        n_items=model.n_items,
        n_layers=model.n_layers,
        train_items=train_interactions,
        domain_map=domain_map,
    )


def stacked_initial_embeddings(model: "LightGCN") -> np.ndarray:
    """Layer-0 embeddings as (n_users + n_items, dim), same order as the graph nodes."""
    u = model.user_embedding.weight.detach().cpu().numpy()
    it = model.item_embedding.weight.detach().cpu().numpy()
    return np.concatenate([u, it], axis=0)


def verify_explanation(
    user_id: int,
    explainer: LightGCNExplainer,
    initial_embeddings: np.ndarray,
    model_final_embedding: np.ndarray,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Check w @ E0 ≈ E_u for the same user, where E_u is the model's final user vector.
    """
    weights = explainer.get_raw_weights(user_id)
    reconstructed = weights @ initial_embeddings
    diff = np.abs(reconstructed - model_final_embedding)
    return {
        "match": bool(np.allclose(reconstructed, model_final_embedding, atol=atol)),
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "reconstructed": reconstructed,
        "model_output": model_final_embedding,
    }


def save_adj_with_meta(
    adj: sp.spmatrix,
    n_users: int,
    n_items: int,
    n_layers: int,
    path: str,
    user_mapping: Optional[Dict] = None,
    item_mapping: Optional[Dict] = None,
) -> None:
    """Save adjacency matrix and its metadata as a sidecar JSON.

    Writes two files:
        <path>       — the scipy sparse matrix (.npz)
        <path>.json  — n_users, n_items, n_layers, and optionally
                       user_mapping / item_mapping {original_id: internal_id}

    user_mapping and item_mapping let explain.py accept original IDs from the
    raw data file and display original IDs in its output.
    """
    p = Path(path)
    sp.save_npz(str(p), adj)
    meta: Dict[str, Any] = {"n_users": n_users, "n_items": n_items, "n_layers": n_layers}
    if user_mapping is not None:
        meta["user_mapping"] = {str(k): v for k, v in user_mapping.items()}
    if item_mapping is not None:
        meta["item_mapping"] = {str(k): v for k, v in item_mapping.items()}
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_adj_with_meta(path: str) -> Tuple[sp.spmatrix, Dict]:
    """Load adjacency matrix and its sidecar metadata JSON.

    Returns:
        adj:  scipy sparse matrix
        meta: dict with n_users, n_items, n_layers, and optionally
              user_mapping / item_mapping {original_id_str: internal_id}
    """
    meta_path = Path(path).with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}\n"
            "Generate it by passing --adj_output during training."
        )
    meta = json.loads(meta_path.read_text())
    return sp.load_npz(path), meta
