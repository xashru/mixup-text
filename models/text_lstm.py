import torch.nn as nn
import torch

embed_size = 300
hidden_size = 100
hidden_layers = 3
bidirectional = True


class TextLSTM(nn.Module):
    def __init__(self, vocab_size, sequence_len, num_class, word_embeddings=None, fine_tune=True, dropout=0.5):
        super(TextLSTM, self).__init__()

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.sequence_len = sequence_len

        if word_embeddings is not None:
            self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=fine_tune)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=hidden_layers, dropout=dropout,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * hidden_layers * (1 + bidirectional), num_class)

    def forward(self, x):
        # (seq_len, batch, embed)
        x = self.embeddings(x)

        _, (x, _) = self.lstm(x)
        # (num_layers * num_directions, batch, hidden)

        x = torch.cat([x[i, :, :] for i in range(x.shape[0])], dim=1)
        # (batch, num_layers * num_directions * hidden_size)
        x = self.fc(x)
        return x

    def _forward_dense(self, x):
        x = self.embeddings(x)
        _, (x, _) = self.lstm(x)
        x = torch.cat([x[i, :, :] for i in range(x.shape[0])], dim=1)
        return x

    def forward_mix_embed(self, x1, x2, lam):
        x1 = self.embeddings(x1)
        x2 = self.embeddings(x2)
        x = lam * x1 + (1.0-lam) * x2
        _, (x, _) = self.lstm(x)
        x = torch.cat([x[i, :, :] for i in range(x.shape[0])], dim=1)
        x = self.fc(x)
        return x

    def forward_mix_sent(self, x1, x2, lam):
        y1 = self.forward(x1)
        y2 = self.forward(x2)
        y = lam * y1 + (1.0-lam) * y2
        return y

    def forward_mix_encoder(self, x1, x2, lam):
        y1 = self._forward_dense(x1)
        y2 = self._forward_dense(x2)
        y = lam * y1 + (1.0-lam) * y2
        y = self.fc(y)
        return y
