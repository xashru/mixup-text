import torch.nn as nn
import torch


embed_size = 300
kernel_size = [3, 4, 5]
num_channels = 100


class TextCNN(nn.Module):
    def __init__(self, vocab_size, sequence_len, num_class, word_embeddings=None, fine_tune=True):
        super(TextCNN, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_size)
        if word_embeddings:
            self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=fine_tune)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_size, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(sequence_len - kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=embed_size, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(sequence_len - kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=embed_size, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(sequence_len - kernel_size[2] + 1)
        )

        self.dropout = nn.Dropout(0.5)

        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels * len(kernel_size), num_class)

    def forward(self, x):
        embedded_sent = self.embeddings(x).permute(1, 2, 0)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)
        return final_out
