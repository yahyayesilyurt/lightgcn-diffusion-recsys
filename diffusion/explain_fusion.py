#!/usr/bin/env python3
"""
Print cross-domain fusion weights (source vs target attention) for c_ud.

Uses the same config and checkpoint paths as train/test:
  cd diffusion && python explain_fusion.py
  python explain_fusion.py --user-id 42   # one internal user id (from user mapping)

Requires: assets, checkpoints, and inters per config.yaml active_dataset.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

from src.config_loader import load_config
from src.dataset import CrossDomainDataset, load_and_pad_embeddings
from src.diffusion_model import ConditionalDiffusion
from src.e2e_wrapper import E2EWrapper


def load_models(checkpoint_path, padded_user_embs, padded_source_embs,
                padded_target_embs, mdl, embed_dim, device):
    e2e_model = E2EWrapper(
        padded_user_embs=padded_user_embs,
        padded_source_embs=padded_source_embs,
        padded_target_embs=padded_target_embs,
        embed_dim=embed_dim,
        num_heads=mdl["num_heads"],
        dropout=mdl["dropout"],
        use_source_stream=mdl["use_source_stream"],
    ).to(device)

    diffusion_model = ConditionalDiffusion(
        steps=mdl["diffusion"]["steps"],
        item_dim=embed_dim,
        cond_dim=embed_dim,
        dropout=mdl["dropout"],
        p_uncond=mdl["diffusion"]["p_uncond"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    e2e_model.load_state_dict(checkpoint["e2e_state_dict"])
    diffusion_model.load_state_dict(checkpoint["diffusion_state_dict"])

    e2e_model.eval()
    diffusion_model.eval()
    return e2e_model, diffusion_model


def main():
    parser = argparse.ArgumentParser(
        description="Print source vs target fusion weights for c_ud"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of dataloader batches to print (ignored if --user-id is set)",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Internal 0-indexed user id (from mappings). Prints fusion weights "
        "for the first training sample that matches this user.",
    )
    args = parser.parse_args()

    cfg = load_config()
    active_ds = cfg["active_dataset"]
    ds_cfg = cfg["datasets"][active_ds]
    paths = ds_cfg["paths"]
    tr = cfg["training"]
    dl = cfg["dataloader"]
    mdl = cfg["model"]
    embed_dim = ds_cfg["model"]["embed_dim"]
    use_source_stream = mdl["use_source_stream"]

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    root = os.path.dirname(os.path.abspath(__file__))

    def _p(key_path):
        p = key_path
        return p if os.path.isabs(p) else os.path.join(root, p)

    emb_path = _p(paths["embeddings"])
    ckpt_path = _p(paths["checkpoints"]["best_model"])

    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    padded_user_embs, padded_source_embs, padded_target_embs = (
        load_and_pad_embeddings(
            pt_file_path=emb_path,
            source_emb_key=ds_cfg["source_emb_key"],
            target_emb_key=ds_cfg["target_emb_key"],
        )
    )

    train_dataset = CrossDomainDataset(
        source_inter_path=_p(paths["inters"]["source_train"]),
        target_inter_path=_p(paths["inters"]["target_train"]),
        source_mapping_path=_p(paths["mappings"]["source"]),
        target_mapping_path=_p(paths["mappings"]["target"]),
        user_mapping_path=_p(paths["mappings"]["user"]),
        max_seq_len=tr["max_seq_len"],
        mode="train",
    )

    if args.user_id is not None:
        match_idx = next(
            (
                i
                for i, s in enumerate(train_dataset.samples)
                if s["user_id"] == args.user_id
            ),
            None,
        )
        if match_idx is None:
            print(
                f"No training sample found for user_id={args.user_id}.",
                file=sys.stderr,
            )
            sys.exit(1)
        loader = DataLoader(
            Subset(train_dataset, [match_idx]),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        args.num_batches = 1
    else:
        loader = DataLoader(
            train_dataset,
            batch_size=dl["train_batch_size"],
            shuffle=False,
            num_workers=dl["train_num_workers"],
            pin_memory=dl["train_pin_memory"],
        )

    e2e_model, _ = load_models(
        ckpt_path,
        padded_user_embs,
        padded_source_embs,
        padded_target_embs,
        mdl,
        embed_dim,
        device,
    )

    print(f"Dataset: {active_ds} | device: {device}")
    if args.user_id is not None:
        print(f"Mode: single sample for user_id={args.user_id} (first match in train set)")
    print(
        "Fusion weights: column 'source' = attention on source intent, "
        "'target' = attention on target intent (cross-attention keys order)."
    )
    print("-" * 60)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            user_ids = batch["user_id"].to(device)
            target_seq = batch["target_seq"].to(device)
            target_mask = batch["target_mask"].to(device)

            if use_source_stream:
                source_seq = batch["source_seq"].to(device)
                source_mask = batch["source_mask"].to(device)
                c_ud, fusion_w = e2e_model(
                    user_ids=user_ids,
                    target_seq_ids=target_seq,
                    target_mask=target_mask,
                    source_seq_ids=source_seq,
                    source_mask=source_mask,
                    return_fusion_weights=True,
                )
            else:
                c_ud, fusion_w = e2e_model(
                    user_ids=user_ids,
                    target_seq_ids=target_seq,
                    target_mask=target_mask,
                    return_fusion_weights=True,
                )

            src_w = fusion_w[:, 0].cpu()
            tgt_w = fusion_w[:, 1].cpu()
            print(f"Batch {i} | batch_size={user_ids.shape[0]} | c_ud shape={tuple(c_ud.shape)}")
            for j in range(user_ids.shape[0]):
                print(
                    f"  user_row={j}  user_id={int(user_ids[j].item())}  "
                    f"source={src_w[j]:.4f}  target={tgt_w[j]:.4f}  "
                    f"sum={(src_w[j]+tgt_w[j]).item():.4f}"
                )


if __name__ == "__main__":
    main()
