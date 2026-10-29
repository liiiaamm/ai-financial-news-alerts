"""
This module contains the BiLSTM-CRF model for Named Entity Recognition.
"""

# 3pps
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
from typing import Optional


class BiLSTMNER(nn.Module):
    """
    BiLSTM-CRF model for token-level Named Entity Recognition.

    Architecture:
        Embedding (GloVe) → LayerNorm → Dropout
        → BiLSTM₁ → Dropout → BiLSTM₂ → Residual
        → LayerNorm → Dropout → Linear → CRF

    Attr:
        num_tags: Number of NER output classes.
        embedding: Embedding layer initialised from pretrained GloVe vectors.
        emb_norm: LayerNorm applied after the embedding lookup.
        dropout: Shared dropout module used throughout the encoder.
        lstm1: First bidirectional LSTM block.
        lstm2: Second bidirectional LSTM block.
        out_norm: LayerNorm applied after the residual combination.
        classifier: Linear projection from hidden states to tag emissions.
        crf: CRF layer that learns tag transition scores.
        class_log_weights: Optional per-class log-frequency bias for the CRF
            emissions. Registered as a buffer (not a parameter).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_tags: int,
        embedding_matrix: np.ndarray,
        dropout: float = 0.3,
    ) -> None:
        """
        Constructor of the BiLSTMNER class.

        Args:
            vocab_size: Vocabulary size, including PAD and UNK tokens.
            embedding_dim: Dimensionality of word embeddings.
            hidden_dim: Number of hidden units per direction in each LSTM.
            num_tags: Number of NER output classes.
            embedding_matrix: Pretrained embedding matrix of shape
                (vocab_size, embedding_dim).
            dropout: Dropout probability applied throughout the encoder.
                Defaults to 0.3.
        """

        # Call super class constructor
        super().__init__()
        torch.manual_seed(42)

        # Store number of tags for external access
        self.num_tags: int = num_tags

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

        # Linear classifier projecting hidden states to tag emissions
        self.classifier: nn.Linear = nn.Linear(hidden_dim * 2, num_tags)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # CRF layer learns valid tag transition scores
        self.crf: CRF = CRF(num_tags, batch_first=True)

        # Per-class log-frequency bias added to emissions to handle tag imbalance
        self.class_log_weights: Optional[torch.Tensor]
        self.register_buffer("class_log_weights", None)

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

    def _emissions(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """
        Computes per-token CRF emission scores for the full encoder stack.

        Args:
            token_ids: Token id tensor of shape (B, L).
            attention_mask: Boolean mask of shape (B, L).

        Returns:
            Emission tensor of shape (B, L, num_tags).
        """

        lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
        L = token_ids.size(1)

        # Embedding lookup with normalization and dropout
        x = self.dropout(self.emb_norm(self.embedding(token_ids)))

        # Two BiLSTM blocks with residual connection
        h1 = self.dropout(self._run_lstm(self.lstm1, x,  lengths, L))
        h2 = self._run_lstm(self.lstm2, h1, lengths, L)
        h  = self.dropout(self.out_norm(h2 + h1))

        # Linear projection to tag space
        emissions = self.classifier(h)

        # Add optional per-class bias to handle tag-frequency imbalance
        if self.class_log_weights is not None:
            emissions = emissions + self.class_log_weights[None, None]

        return emissions

    def forward(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass returning raw token-level emission scores.

        Args:
            token_ids: Token id tensor of shape (B, L).
            attention_mask: Boolean mask of shape (B, L).

        Returns:
            Emission tensor of shape (B, L, num_tags).
        """

        return self._emissions(token_ids, attention_mask)

    def compute_loss(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        tags: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Computes the CRF negative log-likelihood loss.

        Args:
            token_ids: Token id tensor of shape (B, L).
            attention_mask: Boolean mask of shape (B, L).
            tags: Gold tag tensor of shape (B, L), with -100 at padding positions.

        Returns:
            Scalar loss tensor.
        """

        emissions = self._emissions(token_ids, attention_mask)

        # CRF requires valid tag indices everywhere, so replace -100 with 0
        tags_clean = tags.clone()
        tags_clean[tags == -100] = 0

        return -self.crf(emissions, tags_clean, mask=attention_mask, reduction="mean")

    def predict(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.LongTensor:
        """
        Decodes the most likely tag sequence using CRF Viterbi decoding.

        Args:
            token_ids: Token id tensor of shape (B, L).
            attention_mask: Boolean mask of shape (B, L).

        Returns:
            Predicted tag tensor of shape (B, L).
        """

        self.eval()
        with torch.no_grad():
            emissions = self._emissions(token_ids, attention_mask)
            decoded   = self.crf.decode(emissions, mask=attention_mask)

        # Convert variable-length decoded sequences to a padded batch tensor
        B, L   = token_ids.shape
        result = torch.zeros(B, L, dtype=torch.long, device=token_ids.device)
        for i, seq in enumerate(decoded):
            result[i, :len(seq)] = result.new_tensor(seq)

        return result
