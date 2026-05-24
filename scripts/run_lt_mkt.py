import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import LTMKTDataset, get_dataset_config, load_graph, load_question_concepts, load_question_coverage, load_sequences
from lt_mkt import LTMKT
from scripts.train_lt_mkt import LTMKTTrainer


def run(args):
    root = ROOT
    config = get_dataset_config(root, args.dataset)
    data_root = root / "data"
    question_coverage = load_question_coverage(data_root / config["qmatrix"], config["n_questions"])
    train_sequences = load_sequences(data_root / config["train"], args.seq_len, question_coverage)
    test_sequences = load_sequences(data_root / config["test"], args.seq_len, question_coverage)
    train_loader = DataLoader(LTMKTDataset(train_sequences), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(LTMKTDataset(test_sequences), batch_size=args.batch_size, shuffle=False)
    graphs = load_graph(data_root / config["graph"], config["n_concepts"])
    question_concepts = load_question_concepts(data_root / config["qmatrix"], config["n_questions"], config["n_concepts"])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    question_text_embeddings = None
    if args.question_embeddings:
        question_text_embeddings = torch.load(args.question_embeddings, map_location="cpu")
    model = LTMKT(
        n_questions=config["n_questions"],
        n_concepts=config["n_concepts"],
        n_question_difficulty=config["n_question_difficulty"],
        n_domain_transition=config["n_domain_transition"],
        n_domain_coverage=config["n_domain_coverage"],
        question_dim=args.question_dim,
        hidden_dim=args.hidden_dim,
        response_dim=args.response_dim,
        load_dim=args.load_dim,
        dropout=args.dropout,
        question_text_embeddings=question_text_embeddings,
    ).to(device)
    trainer = LTMKTTrainer(model, {key: value.to(device) for key, value in graphs.items()}, question_concepts.to(device), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.dry_run:
        batch = next(iter(train_loader))
        batch = [item.to(device) for item in batch]
        loss, count, _, _ = trainer.step(batch)
        print(json.dumps({"dataset": args.dataset, "loss": float(loss.item()), "count": count}, indent=2))
        return
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(optimizer, train_loader)
        metrics = trainer.evaluate(test_loader)
        best_auc = max(best_auc, metrics["auc"])
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "loss": metrics["loss"],
                    "auc": metrics["auc"],
                    "acc": metrics["acc"],
                    "rmse": metrics["rmse"],
                    "best_auc": best_auc,
                },
                indent=2,
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(description="train LT-MKT")
    parser.add_argument("--dataset", default="JuniorH", choices=["JuniorH", "SeniorH", "PTADiscJP", "PTADiscDS"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--question-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=90)
    parser.add_argument("--response-dim", type=int, default=50)
    parser.add_argument("--load-dim", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--question-embeddings", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
