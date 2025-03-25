import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConvGLU(nn.Module):
    """
    One causal convolution block equipped with GLU activation and weight normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2):
        super(CausalConvGLU, self).__init__()
        # Causal convolution
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels * 2, kernel_size, stride=stride, padding=padding)
        )
        self.glu = nn.GLU(dim=1)  # Gated Linear Unit splits input into two channels for gating

    def forward(self, x):
        """
        Input shape: (batch_size, channels, sequence_length)
        Output shape: (batch_size, channels, sequence_length)
        """
        x = self.conv(x)[:, :, :-2]  # Trim padding for causal effect
        x = self.glu(x)
        return x

class AttnDecoderCausal(nn.Module):
    """
    Multi-block convolutional attention-based decoder using causal convolutions, GLU, and pre-aware coverage attention.
    """
    def __init__(self, hidden_size, output_size, num_blocks=3, kernel_size=3):
        super(AttnDecoderCausal, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_blocks = num_blocks

        # Embedding layer to map input tokens
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(0.3)

        # Decoder blocks with causal convolutions
        self.decoder_blocks = nn.ModuleList([
            CausalConvGLU(hidden_size, hidden_size, kernel_size=kernel_size)
            for _ in range(num_blocks)
        ])

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # Intermediate FC layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output FC layer

        # Attention components
        self.attn_weight = nn.Linear(hidden_size, hidden_size)
        self.context_weight = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)  # Attention scoring

        # Pre-aware attention weights
        self.preaware_w1 = nn.Linear(hidden_size, hidden_size)
        self.preaware_w2 = nn.Linear(hidden_size, hidden_size)

    def preaware_attention(self, decoder_state, encoder_outputs, previous_attention, Wpa):
        """
        Implements pre-aware coverage attention.
        :param decoder_state: Current decoder hidden state (batch, hidden_size)
        :param encoder_outputs: Encoder outputs (batch, seq_len, hidden_size)
        :param previous_attention: Coverage of past attention (batch, seq_len)
        :param Wpa: Pre-aware attention matrix
        """
        batch_size, seq_len, hidden_size = encoder_outputs.size()

        # Mapping function P(h_t)
        context_key = self.preaware_w1(torch.matmul(Wpa, decoder_state))
        current_key = self.preaware_w2(decoder_state)
        preaware_weight = context_key + current_key + decoder_state

        # Attention score calculation
        attention_score = self.v(torch.tanh(
            self.attn_weight(preaware_weight.unsqueeze(1)) + self.context_weight(encoder_outputs)
        )).squeeze(2)

        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_score, dim=1)

        # Combine encoder outputs with attention weights to produce context
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        # Update attention coverage
        updated_attention = previous_attention + attention_weights

        return context, updated_attention

    def forward(self, input_tokens, encoder_outputs, previous_attention, Wpa):
        """
        Forward pass through the decoder.
        :param input_tokens: Input token sequence (batch_size, seq_len)
        :param encoder_outputs: Encoder outputs (batch, seq_len, hidden_size)
        :param previous_attention: Cumulative attention weights (batch, seq_len)
        :param Wpa: Pre-aware attention matrix
        """
        batch_size, seq_len = input_tokens.size()

        # Embedding layer
        embedded = self.embedding(input_tokens)  # Shape: (batch, seq_len, hidden_size)
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(1, 2)  # Shape: (batch, hidden_size, seq_len)

        # Pass through decoder blocks
        x = embedded
        for block in self.decoder_blocks:
            x = block(x)

        x = x.transpose(1, 2)  # Back to (batch, seq_len, hidden_size)

        # Calculate pre-aware attention for each step
        outputs = []
        updated_attention = previous_attention

        for t in range(seq_len):
            decoder_state = x[:, t, :]  # (batch, hidden_size)
            context, updated_attention = self.preaware_attention(
                decoder_state, encoder_outputs, updated_attention, Wpa
            )
            combined = context + decoder_state  # Combine context and current state

            # Pass through FC layers
            fc1_out = F.relu(self.fc1(combined))
            output = F.log_softmax(self.fc2(fc1_out), dim=1)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, output_size)
        return outputs, updated_attention
