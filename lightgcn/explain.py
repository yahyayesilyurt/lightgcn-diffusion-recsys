"""
CLI: explain a user's final LightGCN embedding by graph propagation weights.

Normal usage (adj + metadata generated at training time):
    python explain.py --adj_path adj.npz --user_id <original_user_id>

--user_id is the raw ID exactly as it appears in the data file.
Output item IDs are also the original IDs from the data file.

The .npz and its sidecar .json are produced by passing --adj_output to train.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from explainer import (
    LightGCNExplainer,
    load_adj_with_meta,
    stacked_initial_embeddings,
    verify_explanation,
)
from models import LightGCN
from train import load_dataset


def _resolve_user_id(
    raw: str,
    user_mapping: Optional[Dict[str, int]],
) -> int:
    """
    Resolve --user_id to an internal integer.

    Tries original-ID lookup first (when mapping is available), then falls
    back to parsing raw as an integer (internal ID).
    """
    if user_mapping and raw in user_mapping:
        return user_mapping[raw]
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(
            f"User ID '{raw}' not found in the mapping and is not an integer.\n"
            "Pass the original user ID exactly as it appears in the data file."
        )


def _print_explanation(
    result: dict,
    item_id_map: Optional[Dict[int, str]] = None,
    user_id_map: Optional[Dict[int, str]] = None,
    layerwise: Optional[list] = None,
) -> None:
    uid_internal = result["user_id"]
    uid_display = user_id_map[uid_internal] if user_id_map else str(uid_internal)

    print(f"=== User {uid_display} — embedding explanation ({result['n_layers']} layers) ===\n")
    print(
        f"  Self-contribution: weight={result['self_weight']:.6f} "
        f"({result['self_pct']:.2f}%)  (total weight sum={result['total_weight_sum']:.6f})"
    )

    def item_label(iid: int) -> str:
        return item_id_map[iid] if item_id_map else str(iid)

    def user_label(uid: int) -> str:
        return user_id_map[uid] if user_id_map else str(uid)

    print("\n  Top item contributors:")
    print(f"    {'Rank':<6}{'Item ID':<24}{'direct?':<10}{'weight':<14}{'%':<8}")
    for rank, (iid, w, pct, is_direct) in enumerate(result["item_contributions"], 1):
        tag = "yes" if is_direct else "no"
        label = item_label(iid)
        print(f"    #{rank:<5}{label:<24}{tag:<10}{w:<14.6f}{pct:.2f}%")

    if result.get("user_contributions"):
        print("\n  Top collaborative user contributors:")
        for rank, (u, w, pct) in enumerate(result["user_contributions"], 1):
            print(f"    #{rank:<5}{user_label(u):<24}{w:<14.6f}{pct:.2f}%")

    if "domain_summary" in result:
        print("\n  Domain summary (%):")
        for d, p in result["domain_summary"].items():
            print(f"    {d:<20} {p:.2f}%")

    if layerwise:
        print("\n" + "=" * 60)
        print("  Layer-by-layer breakdown:")
        for entry in layerwise:
            k = entry["layer"]
            print(f"\n  Layer {k}  ({entry['hop_type']})")
            if "self_weight" in entry:
                print(f"    Self: {entry['self_weight']:.6f}  ({entry['self_pct']:.2f}%)")
            for iid, w, pct, d in entry["top_items"][:5]:
                print(
                    f"      {item_label(iid)}: w={w:.6f}  {pct:.2f}%  "
                    f"({'direct' if d else 'indirect'})"
                )
            for u, w, pct in entry["top_users"][:5]:
                print(f"      {user_label(u)}: w={w:.6f}  {pct:.2f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="LightGCN post-hoc embedding explanation")

    p.add_argument("--adj_path", type=str, required=True,
                   help="Path to pre-built adjacency matrix (.npz). "
                        "A sidecar .json must exist alongside it "
                        "(produced by train.py --adj_output).")
    p.add_argument("--user_id", type=str, required=True,
                   help="User ID exactly as it appears in the data file.")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--layerwise", action="store_true")

    # optional override (read from sidecar by default)
    p.add_argument("--n_layers", type=int, default=None,
                   help="Override n_layers from metadata (must match training)")

    # optional: dataset for is_direct annotation only
    p.add_argument("--dataset_path", type=str, default=None,
                   help="Training .inter file; used only to mark direct vs indirect items")
    p.add_argument("--test_path", type=str, default=None)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--split_strategy", type=str, default="leave_one_out",
                   choices=["random", "leave_one_out", "all_train"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rows", type=int, default=None)

    # optional: checkpoint for --verify only
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Model checkpoint (.pt); required only for --verify")
    p.add_argument("--verify", action="store_true",
                   help="Reconstruct embedding from weights and compare to model output "
                        "(requires --checkpoint)")

    args = p.parse_args()

    if args.verify and not args.checkpoint:
        p.error("--verify requires --checkpoint")

    # Load adjacency + metadata (includes user/item mappings when available)
    adj, meta = load_adj_with_meta(args.adj_path)
    n_users: int = meta["n_users"]
    n_items: int = meta["n_items"]
    n_layers: int = args.n_layers if args.n_layers is not None else meta["n_layers"]

    # original_id (str) → internal_id (int)
    user_mapping: Optional[Dict[str, int]] = meta.get("user_mapping")
    item_mapping: Optional[Dict[str, int]] = meta.get("item_mapping")

    # internal_id (int) → original_id (str)  — for display
    item_id_map: Optional[Dict[int, str]] = (
        {v: k for k, v in item_mapping.items()} if item_mapping else None
    )
    user_id_map: Optional[Dict[int, str]] = (
        {v: k for k, v in user_mapping.items()} if user_mapping else None
    )

    # Resolve --user_id to internal integer
    internal_user_id = _resolve_user_id(args.user_id, user_mapping)
    if internal_user_id < 0 or internal_user_id >= n_users:
        raise SystemExit(
            f"Resolved internal user ID {internal_user_id} is out of range "
            f"[0, {n_users}). Check that --user_id matches the training data."
        )

    print(f"Adjacency loaded: n_users={n_users}, n_items={n_items}, n_layers={n_layers}")
    if user_mapping:
        print(f"Resolved user '{args.user_id}' → internal id {internal_user_id}")

    # Load train_interactions only for is_direct tagging
    train_interactions = {}
    if args.dataset_path:
        ns = argparse.Namespace(
            dataset_path=args.dataset_path,
            test_path=args.test_path,
            test_ratio=args.test_ratio,
            split_strategy=args.split_strategy,
            seed=args.seed,
            max_rows=args.max_rows,
        )
        dataset = load_dataset(ns)
        if dataset.n_users != n_users or dataset.n_items != n_items:
            raise SystemExit(
                f"Dataset size mismatch: dataset n_users={dataset.n_users} "
                f"n_items={dataset.n_items} vs metadata n_users={n_users} "
                f"n_items={n_items}"
            )
        train_interactions = dataset.train_interactions

    explainer = LightGCNExplainer(
        adj_mat=adj,
        n_users=n_users,
        n_items=n_items,
        n_layers=n_layers,
        train_items=train_interactions,
    )

    res = explainer.explain_user(internal_user_id, top_k=args.top_k)
    lw = None
    if args.layerwise:
        lw = explainer.explain_user_layerwise(
            internal_user_id, top_k_per_layer=min(args.top_k, 10)
        )
    _print_explanation(res, item_id_map=item_id_map, user_id_map=user_id_map, layerwise=lw)

    if args.verify:
        device = torch.device("cpu")
        checkpoint = torch.load(
            Path(args.checkpoint).resolve(), map_location=device, weights_only=False
        )
        state = checkpoint["model_state_dict"]
        cfg = checkpoint.get("config", {})
        model = LightGCN(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=cfg.get("embedding_dim", state["user_embedding.weight"].shape[1]),
            n_layers=n_layers,
        )
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            user_emb_all, _ = model.get_embeddings()
            e_final = user_emb_all[internal_user_id].cpu().numpy()
        e0 = stacked_initial_embeddings(model)
        v = verify_explanation(internal_user_id, explainer, e0, e_final, atol=1e-3)
        print("\n--- Verification (weights @ E0 vs model final user embedding) ---")
        print(f"  match (allclose atol=1e-3): {v['match']}")
        print(f"  max |diff|: {v['max_diff']:.6e}   mean |diff|: {v['mean_diff']:.6e}")


if __name__ == "__main__":
    main()
