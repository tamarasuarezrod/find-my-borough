import torch.nn as nn

class ScoreModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim_1=16, hidden_dim_2=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
