"""
This module contains the shared BiLSTM encoder for joint NER and SA training.
"""

# 3pps
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SharedBiLSTMEncoder(nn.Module):
    """
    Shared BiLSTM encoder with a NER head (token-level) and an SA head
    (sentence-level with attention).

    The combined training loss is defined as:
        L = lambda_ner * L_NER + lambda_sa * L_SA

    Attr:
        hidden_dim: Number of hidden units per LSTM direction.
        num_tags: Number of NER output classes.
        num_sa_classes: Number of sentiment output classes.
        idx2sa: Mapping from class index to sentiment label string.
        sa_class_weights: Per-class weights for the SA cross-entropy loss.
            Registered as a buffer so it moves with the model.
        embedding: Embedding layer initialised from pretrained GloVe vectors.
        dropout: Shared dropout module applied after the embedding lookup.
        lstm: Two-layer shared bidirectional LSTM encoder.
        ner_head: Linear projection from hidden states to NER tag logits.
        W_attention: Linear layer that scores each token hidden state for SA.
        sa_head: Linear projection from context vector to SA class logits.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_tags: int,
        num_sa_classes: int,
        embedding_matrix: np.ndarray,
        dropout: float = 0.3,
        sa_class_weights: Optional[torch.Tensor] = None,
        idx2sa: Optional[dict] = None,
    ) -> None:
        """
        Constructor of the SharedBiLSTMEncoder class.

        Args:
            vocab_size: Vocabulary size, including PAD and UNK tokens.
            embedding_dim: Dimensionality of word embeddings.
            hidden_dim: Number of hidden units per direction in the LSTM.
            num_tags: Number of NER output classes.
            num_sa_classes: Number of sentiment output classes.
            embedding_matrix: Pretrained embedding matrix of shape
                (vocab_size, embedding_dim).
            dropout: Dropout probability applied after the embedding lookup.
                Defaults to 0.3.
            sa_class_weights: Optional per-class weights for the SA loss.
                Defaults to None (uniform weights).
            idx2sa: Mapping from class index to sentiment label string.
                Defaults to None.
        """

        # Call super class constructor
        super().__init__()
        torch.manual_seed(42)

        # Define attributes
        self.hidden_dim:     int          = hidden_dim
        self.num_tags:       int          = num_tags
        self.num_sa_classes: int          = num_sa_classes
        self.idx2sa:         dict         = idx2sa or {}

        # Register SA class weights as a buffer so they move with the model
        if sa_class_weights is not None:
            self.register_buffer("sa_class_weights", sa_class_weights.float())
        else:
            self.register_buffer("sa_class_weights", torch.ones(num_sa_classes, dtype=torch.float))

        # Shared embedding layer initialised from pretrained GloVe vectors
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = True

        # Shared dropout applied after the embedding lookup
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        # Shared two-layer bidirectional LSTM encoder
        self.lstm: nn.LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # NER head: token-level linear projection
        self.ner_head: nn.Linear = nn.Linear(hidden_dim * 2, num_tags)
        nn.init.xavier_uniform_(self.ner_head.weight)
        nn.init.zeros_(self.ner_head.bias)

        # SA head: attention scoring + sentence-level linear projection
        self.W_attention: nn.Linear = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.sa_head:     nn.Linear = nn.Linear(hidden_dim * 2, num_sa_classes)
        nn.init.xavier_uniform_(self.sa_head.weight)
        nn.init.zeros_(self.sa_head.bias)

        return None

    def forward(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        task: str = "both",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Runs the shared encoder and the requested task-specific head(s).

        Args:
            token_ids: Token id tensor of shape (B, L).
            attention_mask: Boolean mask of shape (B, L), where True indicates
                valid tokens and False indicates padding.
            task: Which head(s) to run. One of "ner", "sa", or "both".
                Defaults to "both".

        Returns:
            Tuple containing:
            - ner_logits: tensor of shape (B, L, num_tags), or None.
            - sa_logits: tensor of shape (B, num_sa_classes), or None.
            - attn_weights: tensor of shape (B, L) with SA attention weights,
                or None.
        """

        # Embedding lookup with dropout
        emb = self.dropout(self.embedding(token_ids))

        # Shared BiLSTM with packed sequences to ignore padding
        lengths    = attention_mask.sum(dim=1).cpu().clamp(min=1)
        packed     = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        hidden, _  = pad_packed_sequence(packed_out, batch_first=True, total_length=token_ids.size(1))

        ner_logits = attn_weights = sa_logits = None

        # NER head: token-level projection
        if task in ("ner", "both"):
            ner_logits = self.ner_head(hidden)

        # SA head: masked attention + context vector + projection
        if task in ("sa", "both"):
            scores       = torch.tanh(self.W_attention(hidden)).squeeze(-1)
            scores       = scores.masked_fill(~attention_mask, -1e9)
            attn_weights = torch.softmax(scores, dim=1)
            context      = (attn_weights.unsqueeze(-1) * hidden).sum(dim=1)
            sa_logits    = self.sa_head(context)

        return ner_logits, sa_logits, attn_weights

    def combined_loss(
        self,
        ner_logits: torch.Tensor,
        sa_logits: torch.Tensor,
        ner_targets: torch.LongTensor,
        sa_targets: torch.LongTensor,
        lambda_ner: float,
        lambda_sa: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the weighted joint loss: L = lambda_ner * L_NER + lambda_sa * L_SA.

        Args:
            ner_logits: NER logit tensor of shape (B, L, num_tags).
            sa_logits: SA logit tensor of shape (B, num_sa_classes).
            ner_targets: Gold NER label tensor of shape (B, L), with -100 at
                padding positions.
            sa_targets: Gold sentiment label tensor of shape (B,).
            lambda_ner: Weight assigned to the NER loss term.
            lambda_sa: Weight assigned to the SA loss term.

        Returns:
            Tuple containing:
            - total_loss: weighted sum of both individual losses.
            - ner_loss: token-level NER cross-entropy loss.
            - sa_loss: sentence-level SA cross-entropy loss.
        """

        # Token-level NER loss, ignoring padding positions
        ner_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        ner_loss      = ner_criterion(ner_logits.view(-1, self.num_tags), ner_targets.view(-1))

        # Sentence-level SA loss with optional class weighting
        sa_criterion = nn.CrossEntropyLoss(weight=self.sa_class_weights)
        sa_loss      = sa_criterion(sa_logits, sa_targets)

        total_loss = lambda_ner * ner_loss + lambda_sa * sa_loss

        return total_loss, ner_loss, sa_loss

    def predict_ner(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.LongTensor:
        """
        Predicts NER tag indices for each token in the input sequence.

        Args:
            token_ids: Token id tensor of shape (B, L).
            attention_mask: Boolean mask of shape (B, L).

        Returns:
            Predicted tag index tensor of shape (B, L).
        """

        self.eval()
        with torch.no_grad():
            ner_logits, _, _ = self.forward(token_ids, attention_mask, task="ner")

        return ner_logits.argmax(dim=-1)

    def predict_sa(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> List[str]:
        """
        Predicts the sentiment label for each sequence in the batch.

        Args:
            token_ids: Token id tensor of shape (B, L).
            attention_mask: Boolean mask of shape (B, L).

        Returns:
            List of predicted sentiment label strings, one per sequence.
        """

        self.eval()
        with torch.no_grad():
            _, sa_logits, _ = self.forward(token_ids, attention_mask, task="sa")
            indices = sa_logits.argmax(dim=-1).tolist()

        # Ensure indices is always a list (argmax returns a scalar for batch size 1)
        if isinstance(indices, int):
            indices = [indices]

        return [self.idx2sa.get(i, str(i)) for i in indices]
