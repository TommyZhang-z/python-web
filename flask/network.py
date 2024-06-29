import torch.nn as nn


# Define the SimpleModel
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)
