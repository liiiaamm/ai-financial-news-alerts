"""
Image captioning model — ResNet-50 encoder + LSTM decoder.

Architecture:
  EncoderCNN : ResNet-50 pretrained, layer3/layer4/fc fine-tuned,
               Linear(2048 → hidden_size) + BatchNorm1d output.
  DecoderRNN : spaCy embedding + LSTM conditioned on image features
               (concatenated at every step + hidden state initialisation).
  Full model : ImageCaptioningModel wraps both.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================================
# Encoder
# ============================================================================

class EncoderCNN(nn.Module):
    """ResNet-50 encoder.

    Fine-tunes layer3, layer4 and the new adapter linear.
    Everything else is frozen.

    Args:
        hidden_size: output feature dimension (matches decoder hidden size).
    """

    def __init__(self, hidden_size: int = 512):
        super().__init__()
        torch.manual_seed(42)

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for name, param in resnet.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # Strip the classification head
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.linear = nn.Linear(resnet.fc.in_features, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.01)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224)
        Returns:
            features: (B, hidden_size)
        """
        features = self.resnet(images)
        features = torch.flatten(features, start_dim=1)
        features = self.bn(self.linear(features))
        return features


# ============================================================================
# Decoder
# ============================================================================

class DecoderRNN(nn.Module):
    """LSTM decoder with image-feature conditioning.

    Image features are:
    - Concatenated with word embeddings at every timestep.
    - Used to initialise the LSTM hidden state.

    The embedding layer is initialised from the spaCy embedding matrix

    Args:
        vocab_size           : caption vocabulary size.
        spacy_embeddings_matrix: FloatTensor (vocab_size, spacy_embed_dim).
        hidden_size          : LSTM hidden dim (must equal encoder output dim).
        num_layers           : number of LSTM layers.
    """

    def __init__(
        self,
        vocab_size: int,
        spacy_embeddings_matrix: torch.Tensor,
        hidden_size: int = 512,
        num_layers: int = 1,
    ):
        super().__init__()
        torch.manual_seed(42)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        spacy_embed_dim = spacy_embeddings_matrix.shape[1]
        self.embed = nn.Embedding.from_pretrained(spacy_embeddings_matrix, freeze=False)

        self.lstm = nn.LSTM(
            input_size=spacy_embed_dim + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        image_features: torch.Tensor,
        captions: torch.LongTensor,
    ) -> torch.Tensor:
        """

        Args:
            image_features: (B, hidden_size) — from EncoderCNN.
            captions:       (B, seq_len)     — token ids including <START> and <END>.
        Returns:
            predictions:    (B, seq_len-1, vocab_size)
        """
        # Drop last token (no prediction needed after <END>)
        word_embeddings = self.embed(captions[:, :-1])        # (B, T-1, spacy_dim)
        T = word_embeddings.shape[1]

        # Repeat image features across time
        img_expanded = image_features.unsqueeze(1).repeat(1, T, 1)  # (B, T-1, hidden)
        lstm_inputs = torch.cat((word_embeddings, img_expanded), dim=2)

        # Initialise hidden/cell from image features
        h0 = image_features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        lstm_out, _ = self.lstm(lstm_inputs, (h0, c0))       # (B, T-1, hidden)
        predictions = self.linear(lstm_out)                   # (B, T-1, vocab_size)
        return predictions

    def greedy_decode(
        self,
        image_features: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int = 20,
    ) -> List[int]:
        """Greedy decoding for a single image (batch_size=1).

        Args:
            image_features: (1, hidden_size).
        Returns:
            List of predicted token indices (excluding <START> and <END>).
        """
        device = image_features.device
        h = image_features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.zeros_like(h)

        word_idx = torch.tensor([start_idx], device=device)
        result: List[int] = []

        for _ in range(max_len):
            emb = self.embed(word_idx.unsqueeze(0))              # (1, 1, spacy_dim)
            img_step = image_features.unsqueeze(1)               # (1, 1, hidden)
            lstm_in = torch.cat((emb, img_step), dim=2)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            word_idx = self.linear(out.squeeze(1)).argmax(dim=-1)

            idx = word_idx.item()
            if idx == end_idx:
                break
            result.append(idx)

        return result


# ============================================================================
# Full model
# ============================================================================

class ImageCaptioningModel(nn.Module):
    """Wraps EncoderCNN + DecoderRNN."""

    def __init__(self, encoder: EncoderCNN, decoder: DecoderRNN):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.LongTensor,
    ) -> torch.Tensor:
        features = self.encoder(images)
        return self.decoder(features, captions)
