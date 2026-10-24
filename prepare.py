"""Audit Flickr8k, fit a training-only vocabulary, and cache frozen visual features."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights
import yaml

from models import BOS, EOS, PAD, resnet_encoder

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")


def load_config(path="config.yaml"):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["data_root"] = str(Path(cfg["data_root"]).expanduser().resolve())
    output = Path(cfg["output_dir"])
    cfg["output_dir"] = str((output if output.is_absolute() else path.parent / output).resolve())
    if cfg["random_image_augmentation"] or cfg["image_size"] != 224:
        raise ValueError("This experiment requires fixed 224x224 image preprocessing")
    expected = {
        "A": {"pretrained_encoder": False, "freeze_encoder": False, "decoder": "lstm"},
        "B": {"pretrained_encoder": True, "freeze_encoder": True, "decoder": "lstm"},
        "C": {"pretrained_encoder": True, "freeze_encoder": True, "decoder": "transformer"},
    }
    if cfg["models"] != expected:
        raise ValueError("The model definitions are fixed: A/B LSTM, C Transformer; B/C frozen")
    if cfg["encoder"] != "resnet18" or cfg["pretrained_weights"] != "IMAGENET1K_V1":
        raise ValueError("Only ResNet-18 with ImageNet V1 cached features is supported")
    return cfg


def check_checkpoint_config(cfg, saved):
    # Paths may move; the removed batch profiler never affected training.
    ignored = {"data_root", "output_dir", "profile_batch_sizes"}
    current = {k: v for k, v in cfg.items() if k not in ignored}
    recorded = {k: v for k, v in saved.items() if k not in ignored}
    if current != recorded:
        raise RuntimeError("Config differs from the recorded experiment")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_runtime(cfg, model_name=None, budget_gib=None):
    torch.set_num_threads(cfg["cpu_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these experiments")
    total = torch.cuda.get_device_properties(0).total_memory
    budget = budget_gib if budget_gib is not None else cfg["gpu_memory_gib"][model_name]
    torch.cuda.set_per_process_memory_fraction(min(budget * 1024 ** 3 / total, 0.95), 0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_all(cfg["seed"])
    return torch.device("cuda")


def image_transform(cfg):
    return transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"]),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(cfg["image_mean"], cfg["image_std"]),
    ])


def source_signature(cfg):
    root = Path(cfg["data_root"])
    filenames = ["Flickr8k.token.txt", "Flickr_8k.trainImages.txt",
                 "Flickr_8k.devImages.txt", "Flickr_8k.testImages.txt"]
    hashes = {name: hashlib.sha256((root / "text" / name).read_bytes()).hexdigest()
              for name in filenames}
    return {"sources": hashes, "vocab_min_frequency": cfg["vocab_min_frequency"],
            "max_caption_words": cfg["max_caption_words"], "tokenization": "PTB_lowercase_WordLevel"}


def prepare_text(cfg):
    root = Path(cfg["data_root"])
    signature = source_signature(cfg)
    data_path, tokenizer_path = root / "data.json", root / "tokenizer.json"
    if data_path.exists() and tokenizer_path.exists():
        data = json.loads(data_path.read_text())
        if data["signature"] == signature:
            print("Reusing audited text and training-only vocabulary", flush=True)
            return data
        raise RuntimeError("Prepared text does not match config/source; do not silently reuse it")

    split_files = {"train": "Flickr_8k.trainImages.txt", "val": "Flickr_8k.devImages.txt",
                   "test": "Flickr_8k.testImages.txt"}
    splits = {name: (root / "text" / filename).read_text().splitlines()
              for name, filename in split_files.items()}
    expected = {"train": 6000, "val": 1000, "test": 1000}
    for name, ids in splits.items():
        if len(ids) != expected[name] or len(set(ids)) != len(ids):
            raise RuntimeError(f"Incorrect {name} image-level split")
    if any(set(splits[a]) & set(splits[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
        raise RuntimeError("Image-level split leakage")

    captions = {}
    for line in (root / "text" / "Flickr8k.token.txt").read_text().splitlines():
        caption_id, caption = line.split("\t", 1)
        name, number = caption_id.rsplit("#", 1)
        group = captions.setdefault(name, {})
        if number in group or not caption.strip():
            raise RuntimeError(f"Duplicate/empty caption: {caption_id}")
        group[number] = caption.strip()

    image_ids = sorted(set().union(*map(set, splits.values())))
    physical = {p.name for p in (root / "Flicker8k_Dataset").glob("*.jpg")}
    for name in image_ids:
        if name not in physical or set(captions.get(name, {})) != set("01234"):
            raise RuntimeError(f"Incomplete benchmark image: {name}")
    raw = {name: [captions[name][str(i)] for i in range(5)] for name in image_ids}
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    references = PTBTokenizer().tokenize(
        {name: [{"caption": value} for value in values] for name, values in raw.items()}
    )

    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.WordLevelTrainer(
        min_frequency=cfg["vocab_min_frequency"], vocab_size=100000,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"], show_progress=False,
    )
    tokenizer.train_from_iterator(
        (caption for name in splits["train"] for caption in references[name]), trainer
    )
    assert [tokenizer.token_to_id(t) for t in ("<pad>", "<unk>", "<bos>", "<eos>")] == [0, 1, 2, 3]
    tokenizer.save(str(tokenizer_path))
    index = {name: i for i, name in enumerate(image_ids)}
    records, truncated = [], 0
    for name in splits["train"]:
        for caption in references[name]:
            words = tokenizer.encode(caption).ids
            truncated += len(words) > cfg["max_caption_words"]
            records.append([index[name], [BOS] + words[:cfg["max_caption_words"]] + [EOS]])

    data = {"signature": signature, "image_ids": image_ids, "splits": splits,
            "references": references, "raw_captions": raw, "train_records": records,
            "vocab_size": tokenizer.get_vocab_size(),
            "audit": {"physical_images": len(physical), "used_images": len(image_ids),
                      "unused_images": sorted(physical - set(image_ids)),
                      "caption_ids_without_images": sorted(set(captions) - physical),
                      "training_pairs": len(records), "truncated_training_captions": truncated,
                      "max_training_words_before_truncation": max(
                          len(c.split()) for name in splits["train"] for c in references[name])}}
    write_json(data_path, data)
    print(json.dumps({"vocab_size": data["vocab_size"], "audit": data["audit"]}, indent=2), flush=True)
    return data


class ImageDataset(Dataset):
    def __init__(self, cfg, image_ids):
        self.directory = Path(cfg["data_root"]) / "Flicker8k_Dataset"
        self.image_ids = image_ids
        self.transform = image_transform(cfg)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        name = self.image_ids[index]
        with Image.open(self.directory / name) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, name


def feature_signature(cfg):
    return {key: cfg[key] for key in ("encoder", "pretrained_weights", "image_size", "image_resize",
                                      "image_mean", "image_std", "random_image_augmentation")}


def prepare_features(cfg, data):
    root = Path(cfg["data_root"])
    target = root / cfg["feature_cache"]
    if target.exists():
        saved = torch.load(target, map_location="cpu", mmap=True, weights_only=True)
        if saved["image_ids"] != data["image_ids"] or saved["signature"] != feature_signature(cfg):
            raise RuntimeError("Feature cache does not match images/weights/preprocessing")
        print(f"Reusing visual cache {tuple(saved['features'].shape)}", flush=True)
        return
    device = setup_runtime(cfg, budget_gib=10.0)
    weights = ResNet18_Weights.IMAGENET1K_V1
    state = torch.hub.load_state_dict_from_url(weights.url, model_dir=str(root),
                                             check_hash=True, map_location="cpu")
    weight_path = root / Path(weights.url).name
    del state
    encoder = resnet_encoder(weight_path).eval().requires_grad_(False).to(device)
    encoder = encoder.to(memory_format=torch.channels_last)
    encoder_parameters = sum(p.numel() for p in encoder.parameters())
    loader = DataLoader(ImageDataset(cfg, data["image_ids"]), batch_size=256,
                        num_workers=cfg["num_workers"], pin_memory=True,
                        multiprocessing_context="spawn" if cfg["num_workers"] else None)
    features = torch.empty(len(data["image_ids"]), 512, 7, 7, dtype=torch.float16)
    offset, started = 0, time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for images, _ in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            batch = encoder(images)
            if batch.shape[1:] != (512, 7, 7) or not torch.isfinite(batch).all():
                raise RuntimeError("Invalid frozen encoder output")
            features[offset:offset + len(images)] = batch.cpu().to(torch.float16)
            offset += len(images)
            print(f"feature_cache {offset}/{len(features)}", flush=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save({"features": features, "image_ids": data["image_ids"],
                "signature": feature_signature(cfg), "encoder_parameters": encoder_parameters,
                "weights_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest()}, temporary)
    temporary.replace(target)
    print(f"Cached all 8000 images in {time.monotonic() - started:.1f}s; {target.stat().st_size / 1024**2:.1f} MiB", flush=True)


def load_prepared(cfg):
    root = Path(cfg["data_root"])
    data = json.loads((root / "data.json").read_text())
    if data["signature"] != source_signature(cfg):
        raise RuntimeError("Prepared text/config/source mismatch")
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    return data, tokenizer


def load_features(cfg, data, model_name):
    if model_name == "A":
        return None
    cache = torch.load(Path(cfg["data_root"]) / cfg["feature_cache"],
                       map_location="cpu", mmap=True, weights_only=True)
    if cache["image_ids"] != data["image_ids"] or cache["signature"] != feature_signature(cfg):
        raise RuntimeError("Stale feature cache")
    features = cache["features"]
    # Derive B's pooled vectors once; do not write a redundant second cache.
    return features.mean(dim=(2, 3)) if model_name == "B" else features


class CaptionDataset(Dataset):
    def __init__(self, cfg, data, model_name, split="train", training=True, features=None):
        self.training, self.data = training, data
        self.images = ImageDataset(cfg, data["image_ids"]) if model_name == "A" else None
        self.features = features
        lookup = {name: i for i, name in enumerate(data["image_ids"])}
        self.records = data["train_records"] if training else [(lookup[name], name) for name in data["splits"][split]]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        image_index, value = self.records[item]
        visual = self.images[image_index][0] if self.images is not None else self.features[image_index]
        return visual, torch.tensor(value, dtype=torch.long) if self.training else value


def collate_captions(batch):
    images, captions = zip(*batch)
    padded = pad_sequence(captions, batch_first=True, padding_value=PAD)
    return torch.stack(images), padded[:, :-1], padded[:, 1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = prepare_text(cfg)
    if not args.text_only:
        prepare_features(cfg, data)


if __name__ == "__main__":
    main()
