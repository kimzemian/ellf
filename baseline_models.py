import torch.nn as nn
import torch

import torch


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MLP, self).__init__()
        layers = [
            nn.Linear(
                input_size,
                hidden_size,
            ),
            nn.Tanh(),
        ]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            # layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(
                nn.Tanh(),
            )
        layers.append(nn.Linear(hidden_size, output_size, bias=False))
        self.model = nn.Sequential(*layers)

        # === Init ===
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="tanh")

        self.model.apply(init_weights)
        # self.model[-1].bias.data[0] = 2.24

    def forward(self, x):
        return self.model(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(2, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

    def reshape_for_lstm(self, x):
        batch_size, double_seq_len = x.shape
        seq_len = double_seq_len // 2
        A = x[:, :seq_len]
        B = x[:, seq_len:]
        return torch.stack([A, B], dim=2)

    def forward(self, x):
        # Assuming x shape: (batch, seq_len, input_size)
        x = self.reshape_for_lstm(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Use last output for prediction


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_feats,
        hidden_size,
        output_size,
        num_layers,
        max_len=365,
    ):
        super(TransformerModel, self).__init__()
        self.num_feats = num_feats
        self.input_proj = nn.Linear(num_feats, hidden_size)

        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, max_len, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            batch_first=True,
            norm_first=True,
            dim_feedforward=1024,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)  # or use out[:, -1, :] if preferred
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

    def reshape_for_transformer(self, x):
        num_feats = self.num_feats
        batch_size, total_len = x.shape
        seq_len = total_len // num_feats
        chunks = torch.chunk(x, num_feats, dim=1)
        x_reshaped = torch.stack(
            chunks, dim=2
        )  # shape: [batch_size, seq_len, num_feats]
        return x_reshaped

    # def reshape_for_transformer(self, x):
    #     batch_size, double_seq_len = x.shape
    #     seq_len = double_seq_len // self.num_feats
    #     A = x[:, :seq_len]
    #     B = x[:, seq_len:]
    #     return torch.stack([A, B], dim=2)

    def forward(self, x):
        x = self.reshape_for_transformer(x)
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        seq_len = x.size(1)
        x = x + self.positional_embedding[:, :seq_len, :]  # inject position
        x = self.transformer(x)
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.fc(pooled)
