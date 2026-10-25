import torch
import torch.nn as nn
try:
    from torchcrf import CRF
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'torchcrf'. Install it with: pip install pytorch-crf"
    ) from exc


class HybridBiLSTMCRFPOSTagger(nn.Module):
    def __init__(
        self,
        num_chars,
        char_emb_dim,
        char_hidden_dim,
        vocab_size,
        word_emb_dim,
        word_hidden_dim,
        num_classes,
        pretrained_word_embeddings=None,
        char_padding_idx=0,
        word_padding_idx=0,
        dropout=0.1,
        freeze_pretrained_word_embeddings=True,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.char_embeddings = nn.Embedding(num_chars, char_emb_dim, padding_idx=char_padding_idx)
        self.char_lstm = nn.LSTM(
            char_emb_dim,
            char_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.word_embeddings = nn.Embedding(vocab_size, word_emb_dim, padding_idx=word_padding_idx)
        if pretrained_word_embeddings is not None:
            expected_shape = self.word_embeddings.weight.shape
            if pretrained_word_embeddings.shape != expected_shape:
                raise ValueError(
                    "pretrained_word_embeddings shape mismatch: "
                    f"expected {tuple(expected_shape)}, got {tuple(pretrained_word_embeddings.shape)}"
                )
            self.word_embeddings.weight = nn.Parameter(
                pretrained_word_embeddings.detach().clone(),
                requires_grad=not freeze_pretrained_word_embeddings,
            )

        combined_dim = (2 * char_hidden_dim) + word_emb_dim
        self.main_lstm = nn.LSTM(
            combined_dim,
            word_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.emission = nn.Linear(2 * word_hidden_dim, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, sentence_word_ids, sentence_char_ids, word_lengths, sentence_lengths=None):
        if sentence_char_ids.dim() != 3:
            raise ValueError("sentence_char_ids must be 3D tensor [B, S, C]")
        if sentence_word_ids.dim() != 2:
            raise ValueError("sentence_word_ids must be 2D tensor [B, S]")
        if word_lengths.shape != sentence_word_ids.shape:
            raise ValueError("word_lengths shape must match sentence_word_ids shape [B, S]")

        batch_size, seq_len, max_char_len = sentence_char_ids.size()

        flat_chars = sentence_char_ids.reshape(-1, max_char_len)
        flat_lengths = word_lengths.reshape(-1).to(dtype=torch.long, device=sentence_char_ids.device)
        flat_lengths = flat_lengths.clamp(min=1, max=max_char_len)
        char_embs = self.char_embeddings(flat_chars)

        sorted_lengths, sort_idx = torch.sort(flat_lengths, descending=True)
        char_embs_sorted = char_embs.index_select(0, sort_idx)
        packed_chars = nn.utils.rnn.pack_padded_sequence(
            char_embs_sorted,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, (h_n, _) = self.char_lstm(packed_chars)
        char_rep_sorted = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)
        unsort_idx = torch.argsort(sort_idx)
        char_rep = char_rep_sorted.index_select(0, unsort_idx)
        char_features = char_rep.reshape(batch_size, seq_len, -1)

        word_features = self.word_embeddings(sentence_word_ids)
        hybrid_features = torch.cat((char_features, word_features), dim=-1)
        hybrid_features = self.dropout(hybrid_features)

        if sentence_lengths is not None:
            if sentence_lengths.dim() != 1 or sentence_lengths.size(0) != batch_size:
                raise ValueError("sentence_lengths must be 1D tensor with shape [B]")
            sentence_lengths = sentence_lengths.to(dtype=torch.long, device=sentence_char_ids.device)
            sentence_lengths = sentence_lengths.clamp(min=1, max=seq_len)
            packed_tokens = nn.utils.rnn.pack_padded_sequence(
                hybrid_features,
                sentence_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.main_lstm(packed_tokens)
            main_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=seq_len,
            )
        else:
            main_lstm_out, _ = self.main_lstm(hybrid_features)

        main_lstm_out = self.dropout(main_lstm_out)
        return self.emission(main_lstm_out)

    def compute_loss(self, emissions, gold_tags, mask):
        if mask.dtype != torch.bool:
            mask = mask.bool()
        return -self.crf(emissions, gold_tags, mask=mask, reduction="mean")

    def decode(self, emissions, mask):
        if mask.dtype != torch.bool:
            mask = mask.bool()
        return self.crf.decode(emissions, mask=mask)

    def predict(self, sentence_word_ids, sentence_char_ids, word_lengths, sentence_lengths=None):
        self.eval()
        with torch.no_grad():
            emissions = self.forward(
                sentence_word_ids=sentence_word_ids,
                sentence_char_ids=sentence_char_ids,
                word_lengths=word_lengths,
                sentence_lengths=sentence_lengths,
            )
            if sentence_lengths is None:
                mask = torch.ones(
                    emissions.size(0),
                    emissions.size(1),
                    dtype=torch.bool,
                    device=emissions.device,
                )
            else:
                seq_len = emissions.size(1)
                mask = (
                    torch.arange(seq_len, device=emissions.device).unsqueeze(0)
                    < sentence_lengths.unsqueeze(1)
                )
            pred_paths = self.decode(emissions, mask)
        return pred_paths, emissions
