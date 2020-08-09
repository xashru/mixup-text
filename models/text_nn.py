import torch.nn as nn
import torch.nn.functional as F

embed_size = 300


class ShallowNN(nn.Module):
    def __init__(self, vocab_size, sequence_len, num_class, word_embeddings=None, fine_tune=True, dropout=0.5):
        super(ShallowNN, self).__init__()
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.sequence_len = sequence_len
        if word_embeddings is not None:
            self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=fine_tune)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_size, num_class)

    def forward(self, x):
        # (batch, seq_len, embed)
        x = self.embeddings(x).permute(1, 2, 0)
        # (batch, embed)
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)

        x = self.dropout(x)
        # (batch, #class)
        x = self.fc(x)
        return x
