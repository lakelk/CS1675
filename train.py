"""Train and check the three Flickr8k captioning experiments."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from evaluate import (evaluate_model, evaluation_loader, generate_predictions,
                      plot_history, score_predictions)
from models import BOS, PAD, Captioner, sinusoidal_1d, sinusoidal_2d
from prepare import (CaptionDataset, check_checkpoint_config, collate_captions, load_config, load_features,
                     load_prepared, seed_all, setup_runtime, write_json)

GIB = 1024 ** 3


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def memory_stats():
    return {"allocated_gib": torch.cuda.memory_allocated() / GIB,
            "reserved_gib": torch.cuda.memory_reserved() / GIB,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB}


def make_model(name, data, cfg):
    model = Captioner(name, data["vocab_size"], cfg).cuda()
    if name == "A":
        model = model.to(memory_format=torch.channels_last)
    return model


def make_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                             weight_decay=cfg["weight_decay"], fused=True)


def checkpoint_save(path, checkpoint):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def check(cfg, data, tokenizer):
    setup_runtime(cfg, budget_gib=6.8)
    assert len(data["train_records"]) == 30000
    assert len(data["image_ids"]) == 8000
    assert sum(len(ids) for ids in data["splits"].values()) == 8000
    assert data["audit"]["truncated_training_captions"] == 0
    pe = sinusoidal_2d(7, 7, 256).reshape(7, 7, 256)
    assert torch.equal(pe[3, 4, :128], sinusoidal_1d(7, 128)[3])
    assert torch.equal(pe[3, 4, 128:], sinusoidal_1d(7, 128)[4])
    assert torch.unique(pe.reshape(49, 256), dim=0).shape[0] == 49
    seed_all(cfg["seed"])
    a = Captioner("A", data["vocab_size"], cfg)
    seed_all(cfg["seed"])
    b = Captioner("B", data["vocab_size"], cfg)
    assert all(torch.equal(a.decoder.state_dict()[key], value)
               for key, value in b.decoder.state_dict().items())
    del a, b

    for name in ("A", "B", "C"):
        features = load_features(cfg, data, name)
        dataset = CaptionDataset(cfg, data, name, features=features)
        visual, inputs, targets = collate_captions([dataset[i] for i in (0, 1, 5)])
        model = make_model(name, data, cfg).train()
        visual, inputs, targets = visual.cuda(), inputs.cuda(), targets.cuda()
        if name == "A":
            visual = visual.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(visual, inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=PAD)
        assert logits.shape == (*inputs.shape, data["vocab_size"])
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.decoder.embedding.weight.grad is not None
        if name == "A":
            assert model.encoder[0].weight.grad is not None
        else:
            assert model.encoder is None
        if name == "C":
            assert "image_pe" not in dict(model.named_parameters())
            assert model.decoder.visual_projection.weight.grad is not None
            model.eval()
            # Alter the future: past outputs must remain unchanged.
            original = inputs.clone()
            changed = inputs.clone()
            changed[:, 4:] = 7
            with torch.no_grad():
                first = model(visual.float(), original)
                second = model(visual.float(), changed)
            torch.testing.assert_close(first[:, :4], second[:, :4], atol=1e-5, rtol=1e-5)
            # Padding appended after valid tokens must not affect their outputs.
            prefix = original[:, :4]
            padded = F.pad(prefix, (0, 3), value=PAD)
            with torch.no_grad():
                short = model(visual.float(), prefix)
                long = model(visual.float(), padded)
            torch.testing.assert_close(short, long[:, :4], atol=1e-5, rtol=1e-5)
        model.eval()
        with torch.autocast("cuda", dtype=torch.float16):
            generated = model.generate(visual, 5)
        assert generated.shape[0] == 3
        assert not generated.eq(PAD).any() and not generated.eq(BOS).any()
        print(f"CHECK {name}: forward, gradients, decoding and masks passed", flush=True)
        del model, visual, inputs, targets, logits, loss, dataset, features
        if name == "C":
            del first, second, short, long
        gc.collect()
        torch.cuda.empty_cache()

    # Verify the token-weighted accumulation rule independently of stochastic layers.
    linear = nn.Linear(3, 7)
    x, y = torch.randn(6, 3), torch.tensor([4, 5, PAD, 6, 2, 1])
    F.cross_entropy(linear(x), y, ignore_index=PAD).backward()
    expected = [p.grad.clone() for p in linear.parameters()]
    linear.zero_grad(set_to_none=True)
    for start in (0, 2, 4):
        (F.cross_entropy(linear(x[start:start + 2]), y[start:start + 2],
                         ignore_index=PAD, reduction="sum") / 12).backward()
    for p, reference in zip(linear.parameters(), expected):
        torch.testing.assert_close(p.grad * 12 / y.ne(PAD).sum(), reference)
    names = data["splits"]["val"][:8]
    exact = {name: data["references"][name][0] for name in names}
    scores = score_predictions(data, exact, full=True)
    nonsense = score_predictions(data, {name: "zzzz qqqq" for name in names})
    assert scores["CIDEr"] > nonsense["CIDEr"]
    assert all(math.isfinite(value) for value in scores.values())
    print(json.dumps({"CHECK": "five metrics and Java METEOR passed", "reference_self_scores": scores}), flush=True)
    print("All checks passed", flush=True)


def train(name, cfg, resume=False):
    import fcntl
    setup_runtime(cfg, name)
    run_dir = Path(cfg["output_dir"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / "training.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another {name} training process already owns this run") from error
    if not resume and any((run_dir / path).exists() for path in ("last.pt", "metrics.jsonl")):
        raise RuntimeError("Existing experiment: pass --resume instead of overwriting it")
    batch_size = cfg["batch_sizes"][name]
    effective_batch = cfg["effective_batch_size"]
    if effective_batch % batch_size:
        raise ValueError("Effective batch must be an integer multiple of physical batch")
    accumulation = effective_batch // batch_size
    normalizer = effective_batch * (cfg["max_caption_words"] + 1)
    data, tokenizer = load_prepared(cfg)
    features = load_features(cfg, data, name)
    model = make_model(name, data, cfg)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["min_learning_rate"])
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(cfg["seed"])
    dataset = CaptionDataset(cfg, data, name, features=features)
    workers = cfg["num_workers"] if name == "A" else 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator,
                        collate_fn=collate_captions, num_workers=workers, pin_memory=True,
                        persistent_workers=bool(workers),
                        multiprocessing_context="spawn" if workers else None)
    val_loader = evaluation_loader(cfg, data, name, "val", features)
    seed_all(cfg["seed"])
    start_epoch, best_score, best_epoch = 1, -math.inf, 0
    optimizer_steps, skipped_optimizer_steps = 0, 0
    if resume:
        saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        check_checkpoint_config(cfg, saved["config"])
        if saved["model_name"] != name:
            raise RuntimeError("Checkpoint belongs to a different model")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        generator.set_state(saved["loader_rng"])
        start_epoch = saved["epoch"] + 1
        best_score, best_epoch = saved["best_score"], saved["best_epoch"]
        optimizer_steps = saved["optimizer_steps"]
        skipped_optimizer_steps = saved["skipped_optimizer_steps"]
        del saved

    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (Path(__file__).parent / file for file in
                            ("prepare.py", "models.py", "train.py", "evaluate.py", "requirements.txt"))}
    frozen_count = 0
    if name != "A":
        cache = torch.load(Path(cfg["data_root"]) / cfg["feature_cache"],
                           map_location="cpu", mmap=True, weights_only=True)
        frozen_count = cache["encoder_parameters"]
        del cache
    info = {"model": name, "config": cfg, "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "frozen_encoder_parameters": frozen_count, "vocab_size": data["vocab_size"],
            "data_audit": data["audit"], "physical_batch": batch_size,
            "effective_batch": effective_batch, "gradient_accumulation_steps": accumulation,
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "source_sha256": sources,
            "created_utc": utc_now(), "pid": os.getpid(), "resume": resume,
            "determinism": "seeded; cuDNN benchmark/TF32 enabled; not bitwise deterministic"}
    write_json(run_dir / "run_info.json", info)
    status = {"model": name, "pid": os.getpid(), "state": "starting", "epoch": start_epoch,
              "total_epochs": cfg["epochs"], "physical_batch": batch_size,
              "effective_batch": effective_batch, "optimizer_steps": optimizer_steps,
              "skipped_optimizer_steps": skipped_optimizer_steps}

    def update_status(**fields):
        status.update(fields)
        status.update(updated_utc=utc_now(), **memory_stats())
        write_json(run_dir / "status.json", status)

    update_status()
    print(json.dumps(info), flush=True)
    try:
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            epoch_started, last_report = time.monotonic(), time.monotonic()
            epoch_loss = torch.zeros((), device="cuda")
            total_tokens, group_tokens, seen = 0, 0, 0
            lr = optimizer.param_groups[0]["lr"]
            update_status(state="training", epoch=epoch, batches=0, examples=0)
            for batch_index, (visual, inputs, targets) in enumerate(loader, 1):
                valid_tokens = int(targets.ne(PAD).sum())
                seen += len(visual)
                visual = visual.to("cuda", non_blocking=True)
                if name == "A":
                    visual = visual.contiguous(memory_format=torch.channels_last)
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(visual, inputs)
                    loss_sum = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                               ignore_index=PAD, reduction="sum")
                scaler.scale(loss_sum / normalizer).backward()
                epoch_loss += loss_sum.detach()
                group_tokens += valid_tokens
                total_tokens += valid_tokens
                if batch_index % accumulation == 0 or batch_index == len(loader):
                    scaler.unscale_(optimizer)
                    # Exact per-token mean across all accumulated microbatches.
                    correction = normalizer / group_tokens
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scaler.get_scale() >= previous_scale:
                        optimizer_steps += 1
                    else:
                        skipped_optimizer_steps += 1
                    group_tokens = 0
                del logits, loss_sum, visual, inputs, targets
                if time.monotonic() - last_report >= 20 or batch_index == len(loader):
                    current_loss = float(epoch_loss) / total_tokens
                    if not math.isfinite(current_loss):
                        raise RuntimeError("Non-finite training loss")
                    update_status(batches=batch_index, examples=seen, optimizer_steps=optimizer_steps,
                                  skipped_optimizer_steps=skipped_optimizer_steps, train_token_loss=current_loss)
                    print(f"{name} epoch {epoch}/{cfg['epochs']} batch {batch_index}/{len(loader)} "
                          f"examples={seen} loss={current_loss:.4f} optimizer_steps={optimizer_steps}", flush=True)
                    last_report = time.monotonic()
            torch.cuda.synchronize()
            train_seconds = time.monotonic() - epoch_started
            update_status(state="validating", examples=seen)
            val_started = time.monotonic()
            predictions = generate_predictions(model, val_loader, tokenizer, cfg)
            if set(predictions) != set(data["splits"]["val"]):
                raise RuntimeError("Validation did not evaluate every unique validation image")
            score = score_predictions(data, predictions)["CIDEr"]
            if not math.isfinite(score):
                raise RuntimeError("Non-finite validation CIDEr")
            val_seconds = time.monotonic() - val_started
            if score > best_score:
                best_score, best_epoch = score, epoch
                checkpoint_save(run_dir / "best.pt", {"model": model.state_dict(), "config": cfg,
                                "model_name": name, "epoch": epoch, "val_CIDEr": score})
                write_json(run_dir / "val_best_predictions.json", predictions)
            scheduler.step()
            checkpoint_save(run_dir / "last.pt", {"model": model.state_dict(), "config": cfg,
                            "model_name": name, "epoch": epoch, "best_score": best_score,
                            "best_epoch": best_epoch, "optimizer_steps": optimizer_steps,
                            "skipped_optimizer_steps": skipped_optimizer_steps,
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all(), "loader_rng": generator.get_state()})
            row = {"epoch": epoch, "train_token_loss": float(epoch_loss) / total_tokens,
                   "val_CIDEr": score, "best_val_CIDEr": best_score, "best_epoch": best_epoch,
                   "lr": lr, "train_seconds": train_seconds, "val_seconds": val_seconds,
                   "examples_per_second": seen / train_seconds, "physical_batch": batch_size,
                   "effective_batch": effective_batch, "optimizer_steps": optimizer_steps,
                   "skipped_optimizer_steps": skipped_optimizer_steps, **memory_stats()}
            with (run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            update_status(state="epoch_complete", **row)
            print(json.dumps({"model": name, **row}), flush=True)
            plot_history(run_dir)
        update_status(state="testing", epoch=cfg["epochs"], best_epoch=best_epoch, best_val_CIDEr=best_score)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scheduler, scaler, loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        if cfg["automatic_test_evaluation"]:
            details = evaluate_model(model, cfg, data, tokenizer, "test", features, run_dir, full=True)
            update_status(test_scores=details["scores"])
        update_status(state="complete", completed_utc=utc_now())
        print(f"{name}: completed all {cfg['epochs']} epochs; best validation epoch={best_epoch}", flush=True)
    except BaseException as error:
        update_status(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=["A", "B", "C"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.check:
        data, tokenizer = load_prepared(cfg)
        check(cfg, data, tokenizer)
    elif args.model is None:
        parser.error("--model is required unless --check is used")
    else:
        train(args.model, cfg, args.resume)


if __name__ == "__main__":
    main()
