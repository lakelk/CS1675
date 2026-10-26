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
conda create -n CS1675 -c conda-forge python=3.12 pip openjdk=17
conda activate CS1675
pip install -r requirements.txt
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

All models train for 100 epochs with batch size 512. The highest validation CIDEr selects the checkpoint, then test evaluation uses greedy decoding. `--check` is a short correctness test, not training. Use `python train.py --model A --resume` to resume, or `python evaluate.py --model A` to re-evaluate. Replace A with B/C as needed.

Outputs: `output/batch512_100/`. Use `--resume` for an existing run or a new `output_dir` to restart.

## Results

100-epoch runs, batch size 512, seed 42. Best validation checkpoints: A/B/C at epochs 28/35/22. Scores below are from the test set.

| Model | CIDEr | BLEU-1 | BLEU-4 | METEOR | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.1961 | 0.4786 | 0.0970 | 0.1469 | 0.3649 |
| B | 0.4023 | 0.5801 | 0.1682 | 0.1858 | 0.4345 |
| C | 0.4556 | 0.5857 | 0.1782 | 0.1892 | 0.4331 |

These are system comparisons: A/B also differ in encoder trainability, and B/C in pooled versus spatial feature access.

Built with PyTorch/torchvision, Hugging Face Tokenizers, and [pycocoevalcap](https://github.com/salaniz/pycocoevalcap).
