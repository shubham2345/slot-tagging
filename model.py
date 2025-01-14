import torch
import torch.nn as nn

import torch.nn.functional as F


class EnhancedBiLSTMTaggerWithCNNLayers(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, token_vocab, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=token_vocab['<PAD>'])

        # First LSTM layer
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout)

        # First CNN layer
        self.conv1d_1 = nn.Conv1d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        self.layer_norm_cnn1 = nn.LayerNorm(hidden_dim * 2)
        self.dropout_cnn1 = nn.Dropout(dropout)

        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2)
        self.dropout2 = nn.Dropout(dropout)

        # Second CNN layer
        self.conv1d_2 = nn.Conv1d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        self.layer_norm_cnn2 = nn.LayerNorm(hidden_dim * 2)
        self.dropout_cnn2 = nn.Dropout(dropout)

        # Third LSTM layer
        self.lstm3 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
        self.layer_norm3 = nn.LayerNorm(hidden_dim * 2)
        self.dropout3 = nn.Dropout(dropout)

        # Third CNN layer
        self.conv1d_3 = nn.Conv1d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        self.layer_norm_cnn3 = nn.LayerNorm(hidden_dim * 2)
        self.dropout_cnn3 = nn.Dropout(dropout)

        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, token_ids):
      # Embedding Layer
      embeddings = self.embedding(token_ids)  # Shape: (batch_size, seq_length, embedding_dim)

      # First LSTM Layer
      lstm_out1, _ = self.lstm1(embeddings)  # Shape: (batch_size, seq_length, hidden_dim * 2)
      lstm_out1 = self.layer_norm1(lstm_out1)
      lstm_out1 = self.dropout1(lstm_out1)

      # First CNN Layer with Residual Connection
      cnn_input1 = lstm_out1.permute(0, 2, 1)  # Shape: (batch_size, hidden_dim * 2, seq_length)
      conv_out1 = self.conv1d_1(cnn_input1)
      conv_out1 = self.layer_norm_cnn1(conv_out1.permute(0, 2, 1)).permute(0, 2, 1)
      conv_out1 = torch.relu(conv_out1)
      conv_out1 = self.dropout_cnn1(conv_out1)
      conv_out1 = conv_out1.permute(0, 2, 1)  # Transpose back to (batch_size, seq_length, hidden_dim * 2)

      # Pad conv_out1 if its sequence length is smaller than lstm_out1
      if conv_out1.size(1) < lstm_out1.size(1):
          pad_size = lstm_out1.size(1) - conv_out1.size(1)
          conv_out1 = F.pad(conv_out1, (0, 0, 0, pad_size))

      conv_out1 = conv_out1 + lstm_out1  # Residual connection

      # Second LSTM Layer
      lstm_out2, _ = self.lstm2(conv_out1)  # Shape: (batch_size, seq_length, hidden_dim * 2)
      lstm_out2 = self.layer_norm2(lstm_out2)
      lstm_out2 = self.dropout2(lstm_out2)

      # Second CNN Layer with Residual Connection
      cnn_input2 = lstm_out2.permute(0, 2, 1)  # Shape: (batch_size, hidden_dim * 2, seq_length)
      conv_out2 = self.conv1d_2(cnn_input2)
      conv_out2 = self.layer_norm_cnn2(conv_out2.permute(0, 2, 1)).permute(0, 2, 1)
      conv_out2 = torch.relu(conv_out2)
      conv_out2 = self.dropout_cnn2(conv_out2)
      conv_out2 = conv_out2.permute(0, 2, 1)  # Transpose back to (batch_size, seq_length, hidden_dim * 2)

      # Pad conv_out2 if its sequence length is smaller than lstm_out2
      if conv_out2.size(1) < lstm_out2.size(1):
          pad_size = lstm_out2.size(1) - conv_out2.size(1)
          conv_out2 = F.pad(conv_out2, (0, 0, 0, pad_size))

      conv_out2 = conv_out2 + lstm_out2  # Residual connection

      # Third LSTM Layer
      lstm_out3, _ = self.lstm3(conv_out2)  # Shape: (batch_size, seq_length, hidden_dim * 2)
      lstm_out3 = self.layer_norm3(lstm_out3)
      lstm_out3 = self.dropout3(lstm_out3)

      # Third CNN Layer with Residual Connection
      cnn_input3 = lstm_out3.permute(0, 2, 1)  # Shape: (batch_size, hidden_dim * 2, seq_length)
      conv_out3 = self.conv1d_3(cnn_input3)
      conv_out3 = self.layer_norm_cnn3(conv_out3.permute(0, 2, 1)).permute(0, 2, 1)
      conv_out3 = torch.relu(conv_out3)
      conv_out3 = self.dropout_cnn3(conv_out3)
      conv_out3 = conv_out3.permute(0, 2, 1)  # Transpose back to (batch_size, seq_length, hidden_dim * 2)

      # Pad conv_out3 if its sequence length is smaller than lstm_out3
      if conv_out3.size(1) < lstm_out3.size(1):
          pad_size = lstm_out3.size(1) - conv_out3.size(1)
          conv_out3 = F.pad(conv_out3, (0, 0, 0, pad_size))

      conv_out3 = conv_out3 + lstm_out3  # Residual connection

      # Fully Connected Layer
      outputs = self.fc(conv_out3)  # Shape: (batch_size, seq_length, tagset_size)

      return outputs



