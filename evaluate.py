"""Autoregressive validation/test evaluation and training-curve plotting."""

import argparse
import json
from pathlib import Path
import time

from torch.utils.data import DataLoader
import torch

from models import BOS, EOS, PAD, Captioner
from prepare import (CaptionDataset, check_checkpoint_config, load_config, load_features,
                     load_prepared, setup_runtime, write_json)


@torch.inference_mode()
def generate_predictions(model, loader, tokenizer, cfg):
    model.eval()
    predictions = {}
    for visual, names in loader:
        visual = visual.to("cuda", non_blocking=True)
        if model.model_name == "A":
            visual = visual.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            generated = model.generate(visual, cfg["max_caption_words"]).cpu().tolist()
        for name, ids in zip(names, generated):
            clean = []
            for token in ids:
                if token == EOS:
                    break
                if token not in (PAD, BOS):
                    clean.append(token)
            # Keep <unk>: never conceal vocabulary failures in reported scores.
            predictions[name] = tokenizer.decode(clean, skip_special_tokens=False)
    return predictions


def score_predictions(data, predictions, full=False):
    from pycocoevalcap.cider.cider import Cider
    references = {name: data["references"][name] for name in predictions}
    hypotheses = {name: [caption] for name, caption in predictions.items()}
    # Reference strings remain full PTB-tokenized originals, never vocabulary IDs.
    cider, _ = Cider().compute_score(references, hypotheses)
    scores = {"CIDEr": float(cider)}
    if full:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.meteor.meteor import Meteor
        bleu, _ = Bleu(4).compute_score(references, hypotheses, verbose=0)
        rouge, _ = Rouge().compute_score(references, hypotheses)
        meteor = Meteor()
        meteor_score, _ = meteor.compute_score(references, hypotheses)
        del meteor
        scores.update({"BLEU-1": float(bleu[0]), "BLEU-4": float(bleu[3]),
                       "METEOR": float(meteor_score), "ROUGE-L": float(rouge)})
    return scores


def evaluation_loader(cfg, data, model_name, split, features):
    dataset = CaptionDataset(cfg, data, model_name, split=split, training=False, features=features)
    workers = min(2, cfg["num_workers"]) if model_name == "A" else 0
    return DataLoader(dataset, batch_size=cfg["validation_batch_sizes"][model_name],
                      num_workers=workers, pin_memory=True, persistent_workers=bool(workers),
                      multiprocessing_context="spawn" if workers else None)


def evaluate_model(model, cfg, data, tokenizer, split, features, run_dir, full=True):
    loader = evaluation_loader(cfg, data, model.model_name, split, features)
    started = time.monotonic()
    predictions = generate_predictions(model, loader, tokenizer, cfg)
    if set(predictions) != set(data["splits"][split]):
        raise RuntimeError("Incomplete image-level evaluation")
    scores = score_predictions(data, predictions, full=full)
    details = {"model": model.model_name, "split": split, "images": len(predictions),
               "references_per_image": 5, "decoding": "greedy", "beam_size": 1,
               "scores": scores, "seconds": time.monotonic() - started,
               "implementation": "pycocoevalcap 1.2; PTB-tokenized originals; raw toolkit score scale"}
    write_json(Path(run_dir) / f"{split}_predictions.json", predictions)
    write_json(Path(run_dir) / f"{split}_metrics.json", details)
    print(json.dumps(details), flush=True)
    return details


def plot_history(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    epochs = [r["epoch"] for r in rows]
    axes[0].plot(epochs, [r["train_token_loss"] for r in rows])
    axes[0].set(title="Training token cross-entropy", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [r["val_CIDEr"] for r in rows])
    axes[1].set(title="Autoregressive validation CIDEr", xlabel="Epoch", ylabel="CIDEr")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(Path(run_dir) / "learning_curves.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=["A", "B", "C"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_runtime(cfg, args.model)
    run_dir = Path(cfg["output_dir"]) / args.model
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    check_checkpoint_config(cfg, checkpoint["config"])
    if checkpoint["model_name"] != args.model:
        raise RuntimeError("Checkpoint belongs to a different model")
    data, tokenizer = load_prepared(cfg)
    features = load_features(cfg, data, args.model)
    model = Captioner(args.model, data["vocab_size"], cfg).cuda()
    if args.model == "A":
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    evaluate_model(model, cfg, data, tokenizer, args.split, features, run_dir)
    plot_history(run_dir)


if __name__ == "__main__":
    main()
