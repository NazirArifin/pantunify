import torch
import torch.nn as nn

class HybridBiLSTMPOSTagger(nn.Module):
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
        super(HybridBiLSTMPOSTagger, self).__init__()

        # 1. JALUR KARAKTER (Char-Level)
        self.char_embeddings = nn.Embedding(num_chars, char_emb_dim, padding_idx=char_padding_idx)
        # Char-BiLSTM untuk mengekstrak representasi fisik kata
        self.char_lstm = nn.LSTM(char_emb_dim, char_hidden_dim,
                                 batch_first=True, bidirectional=True)

        # 2. JALUR KATA (Word-Level)
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

        # 3. UTAMA: WORD-LEVEL BiLSTM
        # Input utama adalah hasil gabungan (Concat) Jalur Karakter dan Jalur Kata
        # Jalur Karakter menghasilkan 2 * char_hidden_dim (karena Bidirectional)
        combined_dim = (2 * char_hidden_dim) + word_emb_dim

        self.main_lstm = nn.LSTM(combined_dim, word_hidden_dim,
                                 batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # 4. LAPISAN KLASIFIKASI (Linear Classifier to Universal Dependencies Tags)
        self.classifier = nn.Linear(2 * word_hidden_dim, num_classes)

    def forward(self, sentence_word_ids, sentence_char_ids, word_lengths, sentence_lengths=None):
        """
        Args:
            sentence_word_ids: Tensor [Batch_Size, Seq_Len] -> ID Kata dari Kamus Baku
            sentence_char_ids: Tensor [Batch_Size, Seq_Len, Max_Char_Len] -> ID Karakter per kata
            word_lengths: Tensor [Batch_Size, Seq_Len] -> Panjang karakter asli tiap kata (untuk packing)
            sentence_lengths: Optional Tensor [Batch_Size] -> Panjang token valid tiap kalimat
        """
        if sentence_char_ids.dim() != 3:
            raise ValueError("sentence_char_ids must be 3D tensor [B, S, C]")
        if sentence_word_ids.dim() != 2:
            raise ValueError("sentence_word_ids must be 2D tensor [B, S]")
        if word_lengths.shape != sentence_word_ids.shape:
            raise ValueError("word_lengths shape must match sentence_word_ids shape [B, S]")

        batch_size, seq_len, max_char_len = sentence_char_ids.size()

        # --- PROSES JALUR KARAKTER ---
        # Ratakan tensor agar bisa diproses oleh Char-LSTM sekaligus
        flat_chars = sentence_char_ids.reshape(-1, max_char_len)  # [Batch_Size * Seq_Len, Max_Char_Len]
        flat_lengths = word_lengths.reshape(-1).to(dtype=torch.long, device=sentence_char_ids.device)
        flat_lengths = flat_lengths.clamp(min=1, max=max_char_len)
        char_embs = self.char_embeddings(flat_chars)  # [Batch_Size * Seq_Len, Max_Char_Len, Char_Emb_Dim]

        # Eksekusi Char-LSTM dengan packed sequence agar padding tidak mengganggu hidden state akhir.
        sorted_lengths, sort_idx = torch.sort(flat_lengths, descending=True)
        char_embs_sorted = char_embs.index_select(0, sort_idx)
        packed_chars = nn.utils.rnn.pack_padded_sequence(
            char_embs_sorted, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, (h_n, _) = self.char_lstm(packed_chars)

        # Ambil state akhir (h_n) dari forward dan backward LSTM sebagai representasi kata
        # h_n shape: [num_layers * num_directions, batch_size * seq_len, char_hidden_dim]
        char_rep_sorted = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)
        unsort_idx = torch.argsort(sort_idx)
        char_rep = char_rep_sorted.index_select(0, unsort_idx)  # [Batch_Size * Seq_Len, 2 * Char_Hidden_Dim]

        # Kembalikan struktur bentuk ke dimensi kalimat
        char_features = char_rep.reshape(batch_size, seq_len, -1)  # [Batch_Size, Seq_Len, 2 * Char_Hidden_Dim]

        # --- PROSES JALUR KATA ---
        word_features = self.word_embeddings(sentence_word_ids)  # [Batch_Size, Seq_Len, Word_Emb_Dim]

        # --- CONCATENATION (PENGGABUNGAN) ---
        hybrid_features = torch.cat((char_features, word_features), dim=-1)  # [Batch_Size, Seq_Len, Combined_Dim]
        hybrid_features = self.dropout(hybrid_features)

        # --- PROSES UTAMA (Main Word-Level BiLSTM) ---
        if sentence_lengths is not None:
            if sentence_lengths.dim() != 1 or sentence_lengths.size(0) != batch_size:
                raise ValueError("sentence_lengths must be 1D tensor with shape [B]")
            sentence_lengths = sentence_lengths.to(dtype=torch.long, device=sentence_char_ids.device)
            sentence_lengths = sentence_lengths.clamp(min=1, max=seq_len)
            packed_tokens = nn.utils.rnn.pack_padded_sequence(
                hybrid_features, sentence_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.main_lstm(packed_tokens)
            main_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=seq_len
            )
        else:
            main_lstm_out, _ = self.main_lstm(hybrid_features)  # [Batch_Size, Seq_Len, 2 * Word_Hidden_Dim]
        main_lstm_out = self.dropout(main_lstm_out)

        # --- LOGITS OUTPUT ---
        logits = self.classifier(main_lstm_out)  # [Batch_Size, Seq_Len, Num_Classes]
        return logits

    @staticmethod
    def compute_loss(logits, gold_tags, pad_tag_id):
        """
        Hitung loss sequence labeling dengan masking pada token PAD.
        """
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            gold_tags.reshape(-1),
            ignore_index=pad_tag_id,
        )

    def predict(self, sentence_word_ids, sentence_char_ids, word_lengths, sentence_lengths=None):
        """
        Inference helper tanpa alokasi grafik gradien.
        Mengembalikan tuple (pred_tags, logits).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                sentence_word_ids=sentence_word_ids,
                sentence_char_ids=sentence_char_ids,
                word_lengths=word_lengths,
                sentence_lengths=sentence_lengths,
            )
            pred_tags = logits.argmax(dim=-1)
        return pred_tags, logits
