# Flickr8k Image Captioning

CS1675 course project: compare three captioning models.

| Model | ResNet-18 encoder | Decoder |
| --- | --- | --- |
| A | Random initialization, trainable | LSTM |
| B | ImageNet pretrained, frozen | LSTM |
| C | ImageNet pretrained, frozen | Transformer |

All decoders and word embeddings are trained from scratch. B/C share cached visual features; LSTM uses pooled features, Transformer uses 49 spatial tokens.

## Setup and data

Requires Linux/WSL, an NVIDIA CUDA GPU, Python 3.12, and Java (for PTB tokenization and METEOR scoring).

```bash
conda create -n CS1675 python=3.12 pip
conda activate CS1675
pip install -r requirements.txt
sudo apt install default-jre
```

Download and extract the image and text archives from the [Flickr8k mirror](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k):
- Images: `~/Data/CS1675/Flicker8k_Dataset/*.jpg`
- Captions and split lists: `~/Data/CS1675/text/`

The train/validation/test splits contain 6,000/1,000/1,000 images. Images are resized to 224x224 without augmentation. Text uses PTB tokenization and a training-only vocabulary (minimum frequency 5). Configure paths and training settings in `config.yaml`.

## Run

From the repository directory, run models sequentially:

```bash
python prepare.py
python train.py --check
python train.py --model A
python train.py --model B
python train.py --model C
```

Training runs 100 epochs; the highest validation CIDEr selects the checkpoint, then test evaluation uses greedy decoding. `--check` is a short correctness test, not training. Use `python train.py --model A --resume` to resume, or `python evaluate.py --model A` to re-evaluate. Replace A with B/C as needed.

Outputs: `output/epochs_100/`. Use `--resume` for an existing run or a new `output_dir` to restart. Training batch sizes must divide the effective batch size of 1024.

## Results

100-epoch runs, seed 42, best validation checkpoint.

| Model | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.1944 | 0.4846 | 0.0991 | 0.1389 | 0.3675 |
| B | 0.4005 | 0.5697 | 0.1661 | 0.1848 | 0.4346 |
| C | 0.4372 | 0.5732 | 0.1709 | 0.1910 | 0.4337 |

These are system comparisons: A/B also differ in encoder trainability, and B/C in pooled versus spatial feature access.

Built with PyTorch/torchvision, Hugging Face Tokenizers, and [pycocoevalcap](https://github.com/salaniz/pycocoevalcap).
