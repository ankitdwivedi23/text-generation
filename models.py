"""Top-level model classes.
"""

import torch
import torch.nn as nn

class RNNModule(nn.Module):
    """Baseline model using LSTM

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors tensor of dimension (|word_vectors|, vector_dim)
        hidden_size (int): Number of features in the hidden state at each layer
        num_layers (int): Number of LSTM layers
    """
    def __init__(self, word_vectors, hidden_size, num_layers=1):
        super(RNNModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(word_vectors)
        self.lstm = nn.LSTM(word_vectors.size(1),
                            hidden_size,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, len(word_vectors))
    
    def forward(self, x, prev_state):
        # Shape of x = (batch_size, sequence_length)
        embed = self.embedding(x)                       # (batch_size, sequence_length, embedding_dim)
        output, state = self.lstm(embed, prev_state)    # (batch_size, sequence_length, hidden_size)
        logits = self.fc(output)                        # (batch_size, sequence_length, |word_vectors|)
        return logits, state
        
    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))