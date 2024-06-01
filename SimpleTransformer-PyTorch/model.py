import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSelfAttention(nn.Module):
    """A simple self-attention module."""

    def __init__(self, embed_size, heads=1):
        super(SimpleSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        # Get Q, K, V matrices
        queries = self.queries(query)
        keys = self.keys(key)
        values = self.values(value)

        # Calculate the attention scores
        energy = torch.bmm(queries, keys.transpose(1, 2))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)

        # Get the weighted value vectors
        out = torch.bmm(attention, values)
        out = self.fc_out(out)
        return out


class SimpleTransformerBlock(nn.Module):
    """A simple transformer block."""

    def __init__(self, embed_size):
        super(SimpleTransformerBlock, self).__init__()
        self.attention = SimpleSelfAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        # Add skip connection, followed by LayerNorm
        x = self.norm1(attention + query)

        forward = self.feed_forward(x)
        # Add skip connection, followed by LayerNorm
        out = self.norm2(forward + x)
        return out


class PositionalEncoding(nn.Module):
    """Add positional encoding to the input tensor."""

    def __init__(self, embed_size, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                position = torch.tensor([[pos]], dtype=torch.float32)
                div_term = torch.pow(
                    10000, (2 * (i // 2)) / torch.tensor(embed_size).float()
                )
                self.encoding[pos, i] = torch.sin(position / div_term)
                self.encoding[pos, i + 1] = torch.cos(position / div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, : x.size(1), :].to(x.device)


class SimpleTransformer(nn.Module):
    """A simple transformer model."""

    def __init__(self, embed_size, max_len, output_size):
        super(SimpleTransformer, self).__init__()
        self.embed = nn.Embedding(output_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_len)
        self.transformer_block = SimpleTransformerBlock(embed_size)
        self.fc_out = nn.Linear(embed_size, output_size)

    def forward(self, x):
        embedding = self.embed(x)
        # Add positional encoding
        embedding += self.pos_encoder(embedding)
        transformer_out = self.transformer_block(embedding, embedding, embedding)
        out = self.fc_out(transformer_out)
        return out
