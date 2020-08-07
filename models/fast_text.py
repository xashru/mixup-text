from torch import nn

embed_size = 300
hidden_size = 10


class FastText(nn.Module):
    def __init__(self, vocab_size, sequence_len, num_class, word_embeddings=None, fine_tune=True, dropout=0.5):
        super(FastText, self).__init__()

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.sequence_len = sequence_len
        if word_embeddings is not None:
            self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=fine_tune)

        # Hidden Layer
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Output Layer
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # (batch, seq_len, embed)
        x = self.embeddings(x).permute(1, 0, 2)

        # (batch, hidden_size)
        x = self.fc1(x.mean(1))
        x = self.dropout(x)

        # (batch, num_class)
        x = self.fc2(x)
        return x

    def _forward_dense(self, x):
        x = self.embeddings(x).permute(1, 0, 2)
        x = self.fc1(x.mean(1))
        return x

    def forward_mix_embed(self, x1, x2, lam):
        x1 = self.embeddings(x1).permute(1, 0, 2)
        x2 = self.embeddings(x2).permute(1, 0, 2)
        x = lam * x1 + (1.0 - lam) * x2

        x = self.fc1(x.mean(1))
        x = self.fc2(x)
        return x

    def forward_mix_sent(self, x1, x2, lam):
        y1 = self.forward(x1)
        y2 = self.forward(x2)
        y = lam * y1 + (1.0 - lam) * y2
        return y

    def forward_mix_encoder(self, x1, x2, lam):
        y1 = self._forward_dense(x1)
        y2 = self._forward_dense(x2)
        y = lam * y1 + (1.0 - lam) * y2
        y = self.fc2(y)
        return y
