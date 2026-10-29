"""
This module contains the BiLSTM model for Sentiment Analysis.
"""

# 3pps
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMSA(nn.Module):
    """
    Bidirectional LSTM classifier with attention for sentiment analysis.

    Architecture:
        Embedding (GloVe) → LayerNorm → Dropout
        → BiLSTM₁ → Dropout → BiLSTM₂ → Residual
        → LayerNorm → Dropout → Attention → Linear

    Attr:
        idx2sa: Mapping from class index to sentiment label string.
        embedding: Embedding layer initialised from pretrained GloVe vectors.
        emb_norm: LayerNorm applied after the embedding lookup.
        dropout: Shared dropout module used throughout the encoder.
        lstm1: First bidirectional LSTM block.
        lstm2: Second bidirectional LSTM block.
        out_norm: LayerNorm applied after the residual combination.
        W_attention: Linear layer that scores each token hidden state.
        classifier: Linear projection from context vector to class logits.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        embedding_matrix: np.ndarray,
        dropout: float = 0.3,
        idx2sa: Dict[int, str] = None,
    ) -> None:
        """
        Constructor of the BiLSTMSA class.

        Args:
            vocab_size: Vocabulary size, including PAD and UNK tokens.
            embedding_dim: Dimensionality of word embeddings.
            hidden_dim: Number of hidden units per direction in each LSTM.
            num_classes: Number of sentiment output classes.
            embedding_matrix: Pretrained embedding matrix of shape
                (vocab_size, embedding_dim).
            dropout: Dropout probability applied throughout the encoder.
                Defaults to 0.3.
            idx2sa: Mapping from class index to sentiment label string.
                Defaults to None.
        """

        # Call super class constructor
        super().__init__()
        torch.manual_seed(42)

        # Store label mapping for decoding predictions
        self.idx2sa: Dict[int, str] = idx2sa or {}

        # Embedding layer initialised from pretrained GloVe vectors
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = True

        # Normalization and regularization
        self.emb_norm: nn.LayerNorm = nn.LayerNorm(embedding_dim)
        self.dropout:  nn.Dropout   = nn.Dropout(dropout)

        # Two stacked bidirectional LSTM blocks
        self.lstm1: nn.LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2: nn.LSTM = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Output normalization applied after the residual connection
        self.out_norm: nn.LayerNorm = nn.LayerNorm(hidden_dim * 2)

        # Attention layer: scores each token hidden state with a single scalar
        self.W_attention: nn.Linear = nn.Linear(hidden_dim * 2, 1, bias=False)

        # Sentence classifier projecting the context vector to class logits
        self.classifier: nn.Linear = nn.Linear(hidden_dim * 2, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        return None

    def _run_lstm(
        self,
        lstm: nn.LSTM,
        x: torch.FloatTensor,
        lengths: torch.LongTensor,
        total_length: int,
    ) -> torch.FloatTensor:
        """
        Runs a single BiLSTM over a padded batch, ignoring padding tokens.

        Args:
            lstm: LSTM module to apply.
            x: Input tensor of shape (B, L, D).
            lengths: True sequence lengths of shape (B,).
            total_length: Target padded length L after unpacking.

        Returns:
            Output tensor of shape (B, L, 2H).
        """

        # Pack sequences to skip padding during recurrent computation
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = lstm(packed)

        # Unpack back to padded tensor
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=total_length)

        return output

    def forward(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Forward pass returning class logits and token attention weights.

        Args:
            token_ids: Token id tensor of shape (B, L).
            attention_mask: Boolean mask of shape (B, L), where True indicates
                valid tokens and False indicates padding.

        Returns:
            Tuple containing:
            - logits: class scores of shape (B, num_classes).
            - weights: token attention weights of shape (B, L).
        """

        lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
        L = token_ids.size(1)

        # Embedding lookup with normalization and dropout
        x = self.dropout(self.emb_norm(self.embedding(token_ids)))

        # Two BiLSTM blocks with residual connection
        h1 = self.dropout(self._run_lstm(self.lstm1, x,  lengths, L))
        h2 = self._run_lstm(self.lstm2, h1, lengths, L)
        h  = self.dropout(self.out_norm(h2 + h1))

        # Masked attention: score each token, mask padding, apply softmax
        scores  = torch.tanh(self.W_attention(h)).squeeze(-1)
        scores  = scores.masked_fill(~attention_mask, -1e9)
        weights = torch.softmax(scores, dim=1)

        # Weighted context vector and final classification
        context = (weights.unsqueeze(-1) * h).sum(dim=1)
        logits  = self.classifier(context)

        return logits, weights

    def predict(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> str:
        """
        Predicts the sentiment label for the input sequence.

        Args:
            token_ids: Token id tensor of shape (B, L).
            attention_mask: Boolean mask of shape (B, L).

        Returns:
            Predicted sentiment label as a string.
        """

        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(token_ids, attention_mask)
            idx = int(logits.argmax(dim=-1).item())

        return self.idx2sa.get(idx, str(idx))

    def predict_proba(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> Tuple[str, Dict[str, float], list]:
        """
        Predicts the sentiment label and per-class probabilities.

        Applies adaptive temperature scaling to avoid degenerate softmax
        outputs when logits have a very large or very small range.

        Args:
            token_ids: Token id tensor of shape (B, L).
            attention_mask: Boolean mask of shape (B, L).

        Returns:
            Tuple containing:
            - label: predicted sentiment label as a string.
            - prob_dict: dictionary mapping each class label to its probability.
            - raw_logits: raw logit values as a list, for diagnostics.
        """

        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(token_ids, attention_mask)

            # Scale logits so softmax stays non-degenerate across logit ranges
            logit_range = float((logits.max() - logits.min()).item())
            temperature = max(logit_range / 4.0, 1.0)
            probs = torch.softmax(logits / temperature, dim=-1)[0]

            pred_idx = int(probs.argmax().item())
            label    = self.idx2sa.get(pred_idx, str(pred_idx))

            # Build per-class probability dictionary
            prob_dict = {
                self.idx2sa.get(i, str(i)): round(float(probs[i].item()), 4)
                for i in range(probs.size(0))
            }

        return label, prob_dict, logits[0].tolist()
