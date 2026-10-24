"""ResNet-LSTM baselines and a compact CNN-Transformer captioner."""

import torch
from torch import nn
from torchvision.models import resnet18

PAD, UNK, BOS, EOS = 0, 1, 2, 3


def resnet_encoder(weights_path=None):
    backbone = resnet18(weights=None)
    if weights_path is not None:
        backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return nn.Sequential(*list(backbone.children())[:-2])


def sinusoidal_1d(length, dim, base=10000.0):
    if dim % 2:
        raise ValueError("Sinusoidal dimensions must be even")
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    denominators = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    angles = positions / denominators
    pe = torch.empty(length, dim)
    pe[:, 0::2], pe[:, 1::2] = angles.sin(), angles.cos()
    return pe


def sinusoidal_2d(height, width, dim, base=10000.0):
    if dim % 4:
        raise ValueError("2D sinusoidal dimensions must be divisible by four")
    half = dim // 2
    rows = sinusoidal_1d(height, half, base)[:, None, :].expand(height, width, half)
    cols = sinusoidal_1d(width, half, base)[None, :, :].expand(height, width, half)
    return torch.cat((rows, cols), dim=-1).reshape(height * width, dim)


def choose_next(logits, finished):
    logits[:, PAD] = -torch.inf
    logits[:, BOS] = -torch.inf
    predicted = logits.argmax(dim=-1)
    return torch.where(finished, EOS, predicted)


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        dim, hidden = cfg["embedding_dim"], cfg["lstm_hidden_dim"]
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.init_hidden = nn.Linear(512, hidden)
        self.init_cell = nn.Linear(512, hidden)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.output = nn.Linear(hidden, vocab_size)

    def initial_state(self, features):
        pooled = features if features.ndim == 2 else features.mean(dim=(-2, -1))
        return (self.init_hidden(pooled).tanh().unsqueeze(0),
                self.init_cell(pooled).tanh().unsqueeze(0))

    def forward(self, features, tokens):
        embedded = self.dropout(self.embedding(tokens))
        hidden, _ = self.lstm(embedded, self.initial_state(features))
        return self.output(self.dropout(hidden))

    @torch.inference_mode()
    def generate(self, features, max_words):
        state = self.initial_state(features)
        token = torch.full((features.shape[0],), BOS, device=features.device, dtype=torch.long)
        finished = torch.zeros_like(token, dtype=torch.bool)
        generated = []
        for _ in range(max_words):
            hidden, state = self.lstm(self.embedding(token[:, None]), state)
            token = choose_next(self.output(hidden[:, 0]), finished)
            generated.append(token)
            finished |= token.eq(EOS)
            if finished.all():
                break
        return torch.stack(generated, dim=1)


class TransformerCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        dim = cfg["embedding_dim"]
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.visual_projection = nn.Linear(512, dim)
        self.register_buffer("image_pe", sinusoidal_2d(7, 7, dim, cfg["position_base"])[None])
        self.register_buffer("text_pe", sinusoidal_1d(cfg["max_caption_words"] + 1,
                                                     dim, cfg["position_base"])[None])
        self.dropout = nn.Dropout(cfg["dropout"])
        layer = nn.TransformerDecoderLayer(
            dim, cfg["transformer_heads"], cfg["transformer_ff_dim"],
            dropout=cfg["dropout"], batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerDecoder(layer, cfg["transformer_layers"])
        self.output = nn.Linear(dim, vocab_size)
        # Cloned Transformer layers must not retain identical initial weights.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)

    def memory(self, features):
        projected = self.visual_projection(features.flatten(2).transpose(1, 2))
        return projected + self.image_pe.to(dtype=projected.dtype)

    def hidden(self, tokens, memory):
        length = tokens.shape[1]
        embedded = self.embedding(tokens) + self.text_pe[:, :length]
        causal_mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        return self.transformer(
            self.dropout(embedded), memory, tgt_mask=causal_mask,
            tgt_key_padding_mask=tokens.eq(PAD),
        )

    def forward(self, features, tokens):
        return self.output(self.hidden(tokens, self.memory(features)))

    @torch.inference_mode()
    def generate(self, features, max_words):
        memory = self.memory(features)
        tokens = torch.full((features.shape[0], 1), BOS, device=features.device, dtype=torch.long)
        finished = torch.zeros(features.shape[0], device=features.device, dtype=torch.bool)
        for _ in range(max_words):
            logits = self.output(self.hidden(tokens, memory)[:, -1])
            predicted = choose_next(logits, finished)
            tokens = torch.cat((tokens, predicted[:, None]), dim=1)
            finished |= predicted.eq(EOS)
            if finished.all():
                break
        return tokens[:, 1:]


class Captioner(nn.Module):
    def __init__(self, model_name, vocab_size, cfg):
        super().__init__()
        if model_name not in ("A", "B", "C"):
            raise ValueError(model_name)
        self.model_name = model_name
        # Build A/B's identical decoder before A's random CNN consumes RNG.
        self.decoder = (TransformerCaptionDecoder(vocab_size, cfg) if model_name == "C"
                        else LSTMDecoder(vocab_size, cfg))
        # B/C deliberately have no encoder in the training process.
        self.encoder = resnet_encoder() if model_name == "A" else None

    def features(self, images_or_features):
        if self.encoder is not None:
            return self.encoder(images_or_features)
        if self.model_name == "B" and images_or_features.shape[1:] == (512,):
            return images_or_features
        if images_or_features.shape[1:] != (512, 7, 7):
            raise ValueError("Frozen models require cached [B, 512, 7, 7] features")
        return images_or_features

    def forward(self, images_or_features, tokens):
        return self.decoder(self.features(images_or_features), tokens)

    @torch.inference_mode()
    def generate(self, images_or_features, max_words):
        return self.decoder.generate(self.features(images_or_features), max_words)
