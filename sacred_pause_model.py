# sacred_pause_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TernaryNeuron(nn.Module):
    def __init__(self, threshold_pos=0.5, threshold_neg=-0.5):
        super().__init__()
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg

    def forward(self, x):
        output = torch.zeros_like(x)
        output[x > self.threshold_pos] = 1
        output[x < self.threshold_neg] = -1
        return output

class SacredPauseModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.ternary_neuron = TernaryNeuron()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        score = self.fc2(x)
        decision = self.ternary_neuron(score)
        return decision

class TrainableSacredPauseModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        score = self.fc2(x)
        return score

class SacredPauseLoss(nn.Module):
    def forward(self, raw_output, target):
        return F.mse_loss(raw_output, target)

def train_model():
    data = [
        ([0.9, 0.8, 0.7], 1),
        ([0.1, 0.0, -0.1], 0),
        ([-0.9, -0.8, -0.7], -1),
        ([0.6, 0.7, 0.5], 1),
        ([0.0, 0.1, -0.2], 0),
        ([-0.6, -0.5, -0.7], -1),
        ([0.4, 0.3, 0.5], 0),
        ([0.95, 0.85, 0.9], 1),
        ([-0.4, -0.6, -0.5], -1),
        ([0.2, 0.1, -0.1], 0),
    ]
    inputs = torch.tensor([x for x, _ in data], dtype=torch.float32)
    labels = torch.tensor([[y] for _, y in data], dtype=torch.float32)
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = TrainableSacredPauseModel()
    criterion = SacredPauseLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        total_loss = 0.0
        for batch_inputs, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    eval_model = SacredPauseModel()
    eval_model.load_state_dict(model.state_dict())
    test_input = torch.tensor([[0.3, 0.2, 0.1], [0.9, 0.9, 0.8], [-0.6, -0.7, -0.5]], dtype=torch.float32)
    decisions = eval_model(test_input)
    print("Ternary Decisions:", decisions.squeeze().tolist())

if __name__ == "__main__":
    train_model()
