"""
LightGCN training entrypoint.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time
from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from data import RecommendationDataset
from evaluation import FullRankingEvaluator, SampledEvaluator
from models import LightGCN
from training import BPRLoss, CheckpointManager, PairwiseSampler, ResultsLogger, print_training_header, train_epoch
from explainer import save_adj_with_meta, sparse_torch_adj_to_scipy_csr


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(args: argparse.Namespace) -> RecommendationDataset:
    if args.test_path:
        return RecommendationDataset.from_separate_files(
            train_filepath=args.dataset_path,
            test_filepath=args.test_path,
            max_rows=args.max_rows,
        )

    return RecommendationDataset.from_file(
        filepath=args.dataset_path,
        test_ratio=args.test_ratio,
        split_strategy=args.split_strategy,
        seed=args.seed,
        max_rows=args.max_rows,
    )


def train_lightgcn(
    dataset: RecommendationDataset,
    config: Dict,
    device: torch.device,
    dataset_name: str,
    dataset_path: str,
    checkpoint_path: str | None = None,
) -> LightGCN:
    print_training_header(
        model_name="LightGCN",
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_train=sum(len(items) for items in dataset.train_interactions.values()),
        n_test=sum(len(items) for items in dataset.test_interactions.values()),
    )

    logger = ResultsLogger()
    checkpoint_mgr = CheckpointManager(
        model_name="LightGCN",
        dataset_name=dataset_name,
        config=config,
    )

    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=config["embedding_dim"],
        n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.0),
        reg_weight=config["reg_weight"],
    ).to(device)
    model.set_adjacency_matrix(dataset.interaction_pairs, device)

    if config.get("adj_output"):
        adj_scipy = sparse_torch_adj_to_scipy_csr(model.adj)
        save_adj_with_meta(
            adj_scipy,
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            n_layers=config["n_layers"],
            path=config["adj_output"],
            user_mapping=dataset.user_mapping,
            item_mapping=dataset.item_mapping,
        )
        print(f"[SAVED] Adjacency + metadata: {config['adj_output']}")

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = BPRLoss()
    sampler = PairwiseSampler(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        train_interactions=dataset.train_interactions,
        batch_size=config["batch_size"],
        n_neg=config.get("n_neg", 1),
        max_samples=config.get("max_samples_per_epoch", 0),
        exclude_positive_negatives=config.get("exclude_positive_negatives", True),
    )

    if config.get("use_sampled_eval", True):
        print("Using Sampled Evaluation (1 pos, 99 neg)...")
        evaluator = SampledEvaluator(
            test_interactions=dataset.test_interactions,
            n_items=dataset.n_items,
            n_negatives=99,
            k_values=config.get("k_values", [10, 20]),
            seed=config.get("seed", 42),
        )
        metric_label = "HR@20"
    else:
        print("Using Full Ranking Evaluation (all items, train interactions masked)...")
        evaluator = FullRankingEvaluator(
            train_interactions=dataset.train_interactions,
            test_interactions=dataset.test_interactions,
            n_items=dataset.n_items,
            k_values=config.get("k_values", [10, 20]),
        )
        metric_label = "Recall@20"

    best_metric = 0.0
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    patience_counter = 0
    start_epoch = 1
    training_start = time.time()
    total_epoch_time = 0.0

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["metric_value"]
        best_epoch = checkpoint["epoch"]
        print(f"[RESUME] Loaded checkpoint from epoch {checkpoint['epoch']}")

    try:
        for epoch in range(start_epoch, config["n_epochs"] + 1):
            start_time = time.time()
            train_metrics = train_epoch(model, sampler, optimizer, loss_fn, device)
            epoch_time = time.time() - start_time
            total_epoch_time += epoch_time
            avg_epoch_time = total_epoch_time / (epoch - start_epoch + 1)

            max_epoch_seconds = float(config.get("max_epoch_seconds", 0))
            if max_epoch_seconds > 0 and epoch_time > max_epoch_seconds:
                print(f"\nEpoch {epoch} exceeded max_epoch_seconds={max_epoch_seconds:.1f}s; stopping run.")
                break

            if epoch % config.get("eval_interval", 10) == 0:
                eval_start = time.time()
                eval_metrics = evaluator.evaluate(model, device)
                eval_time = time.time() - eval_start
                total_time = time.time() - training_start

                print(f"\nEpoch {epoch}/{config['n_epochs']} | Train: {epoch_time:.1f}s | Eval: {eval_time:.1f}s | Total: {total_time/60:.1f}min")
                print(f"  Loss: {train_metrics['loss']:.4f}")
                if config.get("use_sampled_eval", True):
                    print(f"  HR@10: {eval_metrics.get('HR@10', 0.0):.4f} | HR@20: {eval_metrics.get('HR@20', 0.0):.4f}")
                    print(f"  N@10:  {eval_metrics.get('NDCG@10', 0.0):.4f} | N@20:  {eval_metrics.get('NDCG@20', 0.0):.4f}")
                else:
                    print(f"  R@10: {eval_metrics.get('Recall@10', 0.0):.4f} | R@20: {eval_metrics.get('Recall@20', 0.0):.4f}")
                    print(f"  N@10: {eval_metrics.get('NDCG@10', 0.0):.4f} | N@20: {eval_metrics.get('NDCG@20', 0.0):.4f}")

                metric_to_track = eval_metrics.get(metric_label, 0.0)
                if checkpoint_mgr.save_if_best(model, metric_to_track, epoch, config, optimizer):
                    print("  [SAVED] New best model!")

                if metric_to_track > best_metric:
                    best_metric = metric_to_track
                    best_epoch = epoch
                    best_metrics = eval_metrics.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= config.get("patience", 10):
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
            else:
                print(f"\rEpoch {epoch}/{config['n_epochs']} | {epoch_time:.1f}s | Avg: {avg_epoch_time:.1f}s", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    training_time = time.time() - training_start
    print("\n" + "-" * 40)
    print("Best Results:")
    if best_metrics:
        for key, value in best_metrics.items():
            print(f"  {key}: {value:.4f}")
        print(f"\nBest {metric_label}: {best_metric:.4f} at epoch {best_epoch}")
    else:
        print("  No evaluation performed during training")
    print(f"Total training time: {training_time/60:.1f} minutes")

    dataset_info = {
        "n_users": dataset.n_users,
        "n_items": dataset.n_items,
        "n_train": sum(len(items) for items in dataset.train_interactions.values()),
        "n_test": sum(len(items) for items in dataset.test_interactions.values()),
    }
    logger.log_result(
        model_name="LightGCN",
        dataset_name=dataset_name,
        dataset_info=dataset_info,
        config=config,
        metrics=best_metrics,
        best_epoch=best_epoch,
        training_time=training_time,
        run_id=checkpoint_mgr.run_id,
        checkpoint_path=checkpoint_mgr.get_best_checkpoint_path() or "",
        dataset_path=dataset_path,
    )

    print(f"[SAVED] Best checkpoint: {checkpoint_mgr.get_best_checkpoint_path()}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGCN")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=lambda x: int(eval(x)) if "**" in x else int(x), default=2**21)
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--reg_weight", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--negratio", type=int, default=1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--split_strategy", type=str, default="leave_one_out", choices=["random", "leave_one_out", "all_train"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_samples_per_epoch", type=int, default=0)
    parser.add_argument("--max_epoch_seconds", type=float, default=0.0)
    parser.add_argument("--adj_output", type=str, default=None,
                        help="Save adjacency matrix + metadata to this .npz path (for explain.py)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    config = {
        "dataset_path": args.dataset_path,
        "test_path": args.test_path or "",
        "embedding_dim": args.embedding_dim,
        "n_layers": args.n_layers,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "reg_weight": args.reg_weight,
        "n_epochs": args.n_epochs,
        "eval_interval": args.eval_interval,
        "k_values": [10, 20],
        "patience": args.patience,
        "n_neg": args.negratio,
        "use_sampled_eval": args.sampled,
        "seed": args.seed,
        "max_samples_per_epoch": args.max_samples_per_epoch,
        "max_epoch_seconds": args.max_epoch_seconds,
        "adj_output": args.adj_output,
    }

    dataset = load_dataset(args)
    dataset_name = f"{Path(args.dataset_path).stem}__{Path(args.test_path).stem}" if args.test_path else Path(args.dataset_path).stem
    train_lightgcn(
        dataset=dataset,
        config=config,
        device=device,
        dataset_name=dataset_name,
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint,
    )
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
